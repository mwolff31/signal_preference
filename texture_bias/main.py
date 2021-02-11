import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision

import numpy as np
import imageio
import glob
from tqdm import tqdm
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import probabilities_to_decision
from utils import resize, get_stim_ref, get_scapes, Stimulus

def run_shape_pref(args, model):

  # get names of incomplete shapes
  incompletes = []
  with open('incomplete.txt', 'r') as fp:
    for x in fp:
      if x!='\n':
        incompletes.append(x[:-1])

  mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
  std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)

  stim_ref = get_stim_ref(args)

  if args.landscape:
    scapes = get_scapes()

  d = './stimuli/style-transfer-preprocessed-512/*'

  cats = glob.glob(d)

  stim_paths = []
  for cat in cats:
    stim_paths+=glob.glob(cat+'/*')

  texture = 0
  shape = 0
  valid = 0
  total = 0

  print('\ngetting pref data...')

  for i in tqdm(range(len(stim_paths))):

    stim_path = stim_paths[i]
    info = stim_path.split('/')[-1].split('.')[0]
    names = info.split('-')
    shape_name = names[0]
    texture_name = names[1]

    cond = stim_ref[shape_name].category not in texture_name

    if args.only_complete:
      cond = cond and (shape_name not in incompletes)

    if cond:

      img = imageio.imread(stim_path)/255.0

      # background interpolation to white
      mask = stim_ref[shape_name].mask
      interp = args.bg_interp*img + (1.-args.bg_interp)
      img = mask*img + (1.-mask)*interp

      # resizing
      assert args.size <= 1
      if args.size < 1:
        s = int(224*args.size)
        img = resize(img, s)
        pad1 = (224-s)//2
        pad2 = pad1+(224-s)%2
        if args.color=='white':
          img = np.pad(img, ((pad1, pad2), (pad1, pad2), (0,0)),
           'constant', constant_values=1)
        if args.color=='black':
          img = np.pad(img, ((pad1, pad2), (pad1, pad2), (0,0)),
           'constant', constant_values=0)

      # add random landscape to background
      if args.landscape:
        mask = stim_ref[shape_name].mask
        scape = scapes[np.random.randint(len(scapes))]
        img = mask*img+(1.-mask)*scape

      img = np.clip(img, 0., 1.)

      # show stimulus
      if args.show_stims:
        plt.imshow(img)
        plt.show()

      img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
      img = (img.float() - mean) / std

      logs = model(img.cuda())
      probs = f.softmax(logs, dim=-1).cpu().detach().numpy().squeeze()

      mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
      pred = mapping.probabilities_to_decision(probs)

      total+=1

      if pred in info:

        valid+=1

        if pred in shape_name:
          shape+=1
        if pred in texture_name:
          texture+=1

  return shape, texture, valid, total

def main(args):
  
  if args.model=='resnet50':
    model = torchvision.models.resnet50(pretrained=True).cuda().eval()
  if args.model=='resnet18':
    model = torchvision.models.resnet18(pretrained=True).cuda().eval()
  if args.model=='alexnet':
    model = torchvision.models.alexnet(pretrained=True).cuda().eval()
  if args.model=='vgg16':
    model = torchvision.models.vgg16_bn(pretrained=True).cuda().eval()

  shape, texture, valid, total = run_shape_pref(args, model)

  print('\nshape pref:', shape/valid)
  print('texture pref:', texture/valid)

  print('\ntotal accuracy:', valid/total)

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--bg_interp', type=float, default=1,
                      help='background interpolation. lower values=more white')
  parser.add_argument('--size', type=float, default=1,
                      help='fraction of original object size. must be <1')
  parser.add_argument('--landscape', action='store_true',
                      help='place objects on a random landscape background')
  parser.add_argument('--show_stims', action='store_true',
                      help='display the stimuli (for testing)')
  parser.add_argument('--only_complete', action='store_true',
                      help='dont use incomplete shapes')
  parser.add_argument('--model', default='resnet50',
                      help='ImageNet-pretrained model name')
  args = parser.parse_args()

  main(args)