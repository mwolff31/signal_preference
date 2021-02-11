import torch
import torch.nn
import torch.nn.functional as f
import torchvision

import numpy as np
from argparse import ArgumentParser
import glob
import json
import imageio
import csv
from tqdm import tqdm

from data import make_datasets, make_cue_conflict
from utils import get_pair_matrix, mkdir, read_patterns, get_pixel_ref
from train import train_model

def cue_conflict_exp(args, model_pth, name_ref, set_id):

  # runs a cue conflict experiment
  # preference calculation:
  # total times predicted in images containing it / total occurences in images

  pair_mat = get_pair_matrix()

  mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).cuda()
  std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).cuda()

  model = torchvision.models.resnet18(pretrained=False, num_classes=len(name_ref)//2)
  checkpoint = torch.load(model_pth)
  new_sd = {}
  sd = checkpoint['state_dict']
  for k, v in sd.items():
    if 'module' in k:
      k = k[len('module.'):]
    new_sd[k] = v

  model.load_state_dict(new_sd)
  model.cuda()
  model.eval()

  idxs = list(name_ref.keys())

  pref_ref = dict(zip(idxs, [0]*len(idxs)))
  occur_ref = dict(zip(idxs, [0]*len(idxs)))

  stimuli = glob.glob('./data/{}/cue_conflict/*'.format(args.exp_name))

  print('doing cue conflict experiment...')

  for i in tqdm(range(len(stimuli))):

    stim = stimuli[i]
    stim_info = stim.split('/')[-1].split('.')[0]
    stim_info = stim_info.split('_')[0]
    id1 = int(stim_info.split('-')[0])
    id2 = int(stim_info.split('-')[1])

    occur_ref[id1]+=1
    occur_ref[id2]+=1

    img = imageio.imread(stim)
    if img.max() > 1:
      img = img / 255

    img = torch.from_numpy(img)
    img = img.float().permute(2,0,1).unsqueeze(0).cuda()
    img = (img-mean)/std

    logs = model(img)
    probs = f.softmax(logs, dim=-1).cpu().detach().numpy()
    pred = np.argmax(probs)

    if id1 in pair_mat[set_id][pred]:
      pref_ref[id1]+=1
    if id2 in pair_mat[set_id][pred]:
      pref_ref[id2]+=1

  keys = list(pref_ref.keys())
  total_pref = {}

  for key in keys:
    total_pref[key] = pref_ref[key]/occur_ref[key]
    
  return total_pref

def aggregate_results(args, name_ref, num_sets):

  # aggregates and averages results from a bunch of json results
  # containing output from cue conflict experiments

  keys = list(name_ref.keys())
  master_pref = {}
  for key in keys:
    master_pref[key] = 0

  for set_id in range(num_sets):
    for run_id in range(args.num_runs):

      result_pth = './results/{}/set_{}/run_{}.json'.format(args.exp_name, set_id, run_id)

      with open(result_pth, 'r') as fp:
          data = json.load(fp)

      for key in list(data.keys()):
        value = data[key]
        master_pref[int(key)]+= value/(num_sets*args.num_runs)

  return master_pref

def main(args):

  print('')
  print('EXPERIMENT NAME:', args.exp_name)
  print('')

  # get refs
  pair_mat = get_pair_matrix()
  name_ref, pattern_ref = read_patterns(args)
  num_patterns = len(pattern_ref)
  pixel_ref = get_pixel_ref(pattern_ref)

  disp_dict = {}
  keys = list(pixel_ref.keys())
  print('PIXELS PER PATTERN')
  for key in keys:
    print(name_ref[key]+': '+str(pixel_ref[key]))

  mkdir('./models')
  mkdir('./models/{}'.format(args.exp_name))
  mkdir('./results')
  mkdir('./results/{}'.format(args.exp_name))

  print('\ncreating cue conflict stimuli...')
  make_cue_conflict(args, name_ref, pattern_ref)
  print('')

  for set_id in range(len(pair_mat)):
    mkdir('./models/{}/set_{}'.format(args.exp_name, set_id))
    mkdir('./results/{}/set_{}'.format(args.exp_name, set_id))

    for run_id in range(args.num_runs):

      print('SET {} | RUN {}'.format(set_id, run_id))

      if set_id == 0 and not args.skip_data_creation and run_id==0:
        print('making datasets...')
        make_datasets(args, name_ref, pattern_ref)

      model_pth = './models/{}/set_{}/run_{}.pth'.format(args.exp_name, set_id, run_id)
      data_dir = './data/{}/set_{}'.format(args.exp_name, set_id)

      if not args.just_cc:
        # train model
        print('training...')
        train_model(args, model_pth, data_dir)

      # do cue conflict experiment and dump result to a json
      pref = cue_conflict_exp(args, model_pth, name_ref, set_id)
      result_name = './results/{}/set_{}/run_{}.json'.format(args.exp_name, set_id, run_id)
      with open(result_name, 'w') as fp:
          json.dump(pref, fp)

      print('')

  master = aggregate_results(args, name_ref, len(pair_mat))

  ids = list(master.keys())
  prefs = np.asarray(list(master.values()))
  idxs = np.argsort(prefs)[::-1]

  ids = np.asarray(ids)[idxs].tolist()
  prefs = prefs[idxs]
  pixels = [pixel_ref[i] for i in ids]
  names = [name_ref[i] for i in ids]

  save_name = 'results/{}/master.csv'.format(args.exp_name)

  with open(save_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['name', 'pref', 'pixels'])
    for i in range(num_patterns):
      writer.writerow([names[i], prefs[i], pixels[i]])

if __name__ == '__main__':

  parser = ArgumentParser()

  # main experiment settings
  parser.add_argument('--exp_name',
                      help='name for experiment')
  parser.add_argument('--pattern_dir',
                      help='subdir in ./patterns that contains 10 features')
  parser.add_argument('--num_runs', type=int, default=5,
                      help='number of runs')
  parser.add_argument('--imgnet_augment', action='store_true',
                      help='whether to do random crops or not (used in the paper)')
  parser.add_argument('--skip_data_creation', action='store_true',
                      help='skip train/val/test training data creation')
  parser.add_argument('--pred_drop', nargs='+', default=[],
                      help='feature names for predictivity')
  parser.add_argument('--pred_drop_val', type=float, default=1.,
                      help='prob of keeping feature. just one feature')
  parser.add_argument('--color_dev', nargs='+', default=[],
                      help='feature names for color deviation')
  parser.add_argument('--color_dev_eps', type=float, default=0.,
                      help='eps for color deviation. just one feature')
  parser.add_argument('--just_cc', action='store_true',
                      help='just do cue conflict exp')

  # misc data settings
  parser.add_argument('--img_size', type=int, default=224,
                      help='image size')
  parser.add_argument('--pattern_size', type=int, default=64,
                      help='pattern/feature size')
  parser.add_argument('--num_ex', nargs='+', type=int, default=[300,100,100],
                      help='number of examples for train/test/val in that order')
  parser.add_argument('--cc_num_ex', type=int, default=100,
                      help='# of ex for each feature combination in cue conflict set')

  parser.add_argument('--milestones', nargs='+', type=int, default=[30,60])

  # torchvision training args
  parser.add_argument('--gpu', default=None)
  parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                      help='model architecture')
  parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                      help='number of data loading workers (default: 4)')
  parser.add_argument('--epochs', default=90, type=int, metavar='N',
                      help='number of total epochs to run')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                      help='manual epoch number (useful on restarts)')
  parser.add_argument('-b', '--batch-size', default=256, type=int,
                      metavar='N',
                      help='mini-batch size (default: 256), this is the total '
                           'batch size of all GPUs on the current node when '
                           'using Data Parallel or Distributed Data Parallel')
  parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                      metavar='LR', help='initial learning rate', dest='lr')
  parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                      help='momentum')
  parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                      metavar='W', help='weight decay (default: 1e-4)',
                      dest='weight_decay')
  parser.add_argument('-p', '--print-freq', default=10, type=int,
                      metavar='N', help='print frequency (default: 10)')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
                      help='path to latest checkpoint (default: none)')
  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                      help='evaluate model on validation set')
  parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                      help='use pre-trained model')
  parser.add_argument('--world-size', default=-1, type=int,
                      help='number of nodes for distributed training')
  parser.add_argument('--rank', default=-1, type=int,
                      help='node rank for distributed training')
  parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                      help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='nccl', type=str,
                      help='distributed backend')
  parser.add_argument('--seed', default=None, type=int,
                      help='seed for initializing training. ')
  parser.add_argument('--multiprocessing-distributed', action='store_true',
                      help='Use multi-processing distributed training to launch '
                           'N processes per node, which has N GPUs. This is the '
                           'fastest way to use PyTorch for either single node or '
                           'multi node data parallel training')
  args = parser.parse_args()

  main(args)