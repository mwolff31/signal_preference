import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import imageio

plt.rcParams["figure.figsize"] = [8,4.5]
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

from resnet import ResNet
from utils import make_splits, acc, get_img, mkdir, corr

def _main(args, split):

  # split up the data according to deviation in an MNIST classifier's features

  # setting data up
  transform = transforms.Compose([transforms.ToTensor()])

  if split=='train' or split=='val':

    dataset = torchvision.datasets.MNIST(args.mnist_path, download=True, train=True, transform=transform)
    n_cls=10

    train_idxs, val_idxs = make_splits(len(dataset), 1)

  if split=='test':
    dataset = torchvision.datasets.MNIST(args.mnist_path, download=True, train=False, transform=transform)

  if split=='train':
    samp = torch.utils.data.SubsetRandomSampler(train_idxs)
  if split=='val':
    samp = torch.utils.data.SubsetRandomSampler(val_idxs)
  if split=='test':
    samp = None

  loader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=100,
    shuffle=False,
    sampler=samp,
    num_workers=4,
    pin_memory=True)

  if split=='train':
    dataset_size = train_idxs.size
  if split=='val':
    dataset_size = val_idxs.size
  if split=='test':
    dataset_size = len(dataset)

  # load model if applicable
  model = ResNet(n_layer=args.n_layer, conv_reduce=args.conv_reduce, n_cls=10, suppress=True).cuda()
  if not args.first:
    ckpt_pth = './models/model.pth'
    model.load_state_dict(torch.load(ckpt_pth))
  model.eval()

  preds = np.zeros((dataset_size,))
  labels = np.zeros((dataset_size,))
  feats = np.zeros((dataset_size,64))
  imgs = np.zeros((dataset_size,28,28))

  start = 0

  # get features and images
  for _, data in enumerate(loader):

    x, y = data
    x = x.cuda()

    with torch.no_grad():
      _, probs, feat = model((x-0.5)/0.5, with_feats=True)
    probs = probs.cpu().detach().numpy()
    label = y.cpu().detach().numpy()

    end = start+x.shape[0]
    preds[start:end] = np.argmax(probs,axis=-1)
    labels[start:end] = label
    feats[start:end, :] = feat.squeeze().cpu().detach().numpy()
    imgs[start:end, :, :] = x.squeeze().cpu().detach().numpy()

    start+=x.shape[0]

  labels = np.int32(labels)

  centers = np.zeros((10,64))
  occur = np.zeros((10,1))

  # getting class centers
  for i in range(dataset_size):

    feat = feats[i, :]
    label = labels[i]
    centers[label, :]+=feat
    occur[label, :]+=1

  centers = centers/occur

  total_idxs = []

  for i in range(10):

    og_idxs = np.where(labels==i)[0]
    cls_imgs = imgs[og_idxs, ...]

    # sort idxs by distance from class center in feature space
    cls_feats = feats[og_idxs, :]
    dists = np.mean(np.sqrt((cls_feats - centers[i, :].reshape(1,-1))**2), axis=-1)
    idxs = np.argsort(dists.squeeze())

    # pull frac that has lowest deviation
    if split=='train':
      stop_idx = int(args.frac*og_idxs.size)
      if stop_idx == 0:
        stop_idx = 1
      idxs = idxs[:stop_idx]
      total_idxs.append(idxs)
      cls_imgs = cls_imgs[idxs, :]

    np.random.shuffle(cls_imgs)

    k = 0
    s = int(args.frac*og_idxs.size)

    # save images within frac deviation
    # will repeat if need be 
    while k < og_idxs.size:

      j = k % idxs.size

      if split=='train':

        mkdir('./data')
        mkdir('./data/{}'.format(int(args.frac*100)))
        mkdir('./data/{}/{}'.format(int(args.frac*100), i))
        path = './data/{}/{}/{}.png'.format(int(args.frac*100), i, k)

      else:
        mkdir('./data')
        mkdir('./data/{}'.format(split))
        mkdir('./data/{}/{}'.format(split, i))
        path = './data/{}/{}/{}.png'.format(split, i, k)

      im = np.uint8(cls_imgs[j, ...]*255)

      if split=='train' or (split in ['val', 'test'] and args.first):
        imageio.imsave(path, im)

      k+=1

  print('{} split of frac {} created'.format(split, args.frac))

def evaluate(args, model, loader, dataset_size):

  # get accuracy from a model on a given dataset (helper)

  preds = np.zeros((dataset_size,))
  labels = np.zeros((dataset_size,))

  start = 0

  for _, data in enumerate(loader):

    x, y = data
    x = x.cuda()

    if args.add_dot:
      for i in range(x.shape[0]):
        x[i, :, y[i]+9, 1] = 1

    if args.just_dot:
      for i in range(x.shape[0]):
        x[i, :, :, :] = 0
        x[i, :, y[i]+9, 1] = 1

    if args.show_data:
      print(y[0].item())
      plt.imshow(get_img(x[0,...]))
      plt.show()

    with torch.no_grad():
      _, probs = model((x-0.5)/0.5)
    probs = probs.cpu().detach().numpy()
    label = y.cpu().detach().numpy()

    end = start+x.shape[0]
    preds[start:end] = np.argmax(probs,axis=-1)
    labels[start:end] = label

    start+=x.shape[0]

  return acc(preds, labels)

def test(args, ckpt_path):

  # get accuracy from a model

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img[0, ...].squeeze().unsqueeze(0))])

  test = torchvision.datasets.ImageFolder(
    './data/test'.format(args.frac), transform=transform)

  n_cls=10

  test_loader = torch.utils.data.DataLoader(
    test, 
    batch_size=100,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

  model = ResNet(n_layer=args.n_layer, conv_reduce=args.conv_reduce, n_cls=10, suppress=True).cuda()
  model.load_state_dict(torch.load(ckpt_path))
  model.eval()

  acc = evaluate(args, model, test_loader, len(test))

  return acc

def generate_graph(args):

  # test models on datasets and generate graphs

  fracs = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
  pcts = []
  accs = []

  for i in tqdm(range(len(fracs))):

    frac = fracs[i]
    clean_acc = 0
    dot_acc = 0
    for run in range(5):

      dot_model_path = './models/mnist_model_{}_{}_with_dot.pth'.format(frac, run+1)
      da = test(args, dot_model_path)
      dot_acc += da/5

    accs.append(dot_acc)

  if not args.just_dot and not args.add_dot:

    fracs = np.asarray(fracs)
    accs = np.asarray(accs)

    xl = np.min(fracs)-0.025
    xr = np.max(fracs)+0.025

    yd = np.min(accs)-0.025
    yu = np.max(accs)+0.025

    cor = corr(fracs, accs)
    m,b = np.polyfit(fracs, accs, 1)

    plt.scatter(fracs, accs, s=100)
    plt.plot([xl, xr], [m*xl+b, m*xr+b], color='black')

    plt.title('Full MNIST', fontsize=18)
    plt.xlabel('Training Data Deviation Segments', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim(xl, xr)
    plt.ylim(yd, yu)
    annotation = plt.annotate('R='+str(round(cor*100)/100)[:5], (0.75,0.8))
    annotation.set_fontsize(14)

    plt.savefig('./clean.pdf', bbox_inches='tight')
    plt.cla()

  elif args.add_dot:

    fracs = np.asarray(fracs)
    accs = np.asarray(accs)

    xl = np.min(fracs)-0.025
    xr = np.max(fracs)+0.025

    yd = np.min(accs)-0.005
    yu = np.max(accs)+0.005

    cor = corr(fracs, accs)
    m,b = np.polyfit(fracs, accs, 1)

    plt.scatter(fracs, accs, s=100)

    plt.title('Binary shiftMNIST', fontsize=18)
    plt.xlabel('Training Data Deviation Segments', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim(xl, xr)
    plt.ylim(yd, yu)

    plt.savefig('./shift.pdf', bbox_inches='tight')
    plt.cla()

  else:

    fracs = np.asarray(fracs)
    accs = np.asarray(accs)

    xl = np.min(fracs)-0.025
    xr = np.max(fracs)+0.025

    yd = np.min(accs)-0.025
    yu = np.max(accs)+0.025

    cor = corr(fracs, accs)
    m,b = np.polyfit(fracs, accs, 1)

    plt.scatter(fracs, accs, s=100)
    plt.plot([xl, xr], [m*xl+b, m*xr+b], color='black')

    plt.title('Only Location-Based Pixel', fontsize=18)
    plt.xlabel('Training Data Deviation Segments', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlim(xl, xr)
    plt.ylim(yd, yu)
    annotation = plt.annotate('R='+str(round(cor*100)/100)[:5], (0.25,0.8))
    annotation.set_fontsize(14)

    plt.savefig('./dot.pdf', bbox_inches='tight')
    plt.cla()

def main(args):

  if args.mode=='graph':
    generate_graph(args)

  else:
    splits = ['train', 'test', 'val']

    for split in splits:
      _main(args, split)

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--mnist_path',
                      help='path to torchvision MNIST dataset')
  parser.add_argument('--n_layer', type=int, default=3,
                      help='number of layers within each ResNet block')
  parser.add_argument('--conv_reduce', action='store_true',
                      help='use conv projection in ResNet')
  parser.add_argument('--frac', type=float, default=1,
                      help='to control amt of deviation. 1 for full, 0 for no dev')
  parser.add_argument('--add_dot', action='store_true',
                      help='add dot next to MNIST digit')
  parser.add_argument('--just_dot', action='store_true',
                      help='only the dot, no MNIST digit')
  parser.add_argument('--first', action='store_true',
                      help='use for to create data splits')
  parser.add_argument('--mode', default='make_data',
                      help='one of "make_data" or "graph"')
  parser.add_argument('--show_data', action='store_true',
                      help='show image data as it is loading')
  args = parser.parse_args()

  main(args)