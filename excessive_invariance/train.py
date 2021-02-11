import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import imageio

from resnet import ResNet
from utils import acc, get_img

cudnn.benchmark = True

def evaluate(args, model, loader, dataset_size):

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

    with torch.no_grad():
      _, probs = model((x-0.5)/0.5)
    probs = probs.cpu().detach().numpy()
    label = y.cpu().detach().numpy()

    end = start+x.shape[0]
    preds[start:end] = np.argmax(probs,axis=-1)
    labels[start:end] = label

    start+=x.shape[0]

  return acc(preds, labels)

def train(args):

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img[0, ...].squeeze().unsqueeze(0))])

  train = torchvision.datasets.ImageFolder(
      './data/{}'.format(int(args.frac*100)), transform=transform)
  val = torchvision.datasets.ImageFolder(
    './data/val', transform=transform)

  train_loader = torch.utils.data.DataLoader(
    train, 
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True)

  val_loader = torch.utils.data.DataLoader(
    val,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True)

  criterion = nn.CrossEntropyLoss()

  model = ResNet(n_layer=args.n_layer, conv_reduce=args.conv_reduce, n_cls=10).cuda()

  milestones = [int(0.6*args.epochs), int(0.8*args.epochs)]

  optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

  best_valacc = 0

  for i in range(args.epochs):

    print('epoch:', i)

    model.train()

    _range = tqdm(train_loader)

    for _, data in enumerate(_range):

      optimizer.zero_grad()

      x, y = data
      x = x.cuda()
      y = y.cuda()

      if args.add_dot:
        for j in range(x.shape[0]):
          x[j, :, y[j]+9, 1] = 1

      if args.just_dot:
        for j in range(x.shape[0]):
          x[j, :, :, :] = 0
          x[j, :, y[j]+9, 1] = 1

      if args.show_data:
        print(y[0].item())
        plt.imshow(get_img(x[0,...]))
        plt.show()

      logs, _ = model((x-0.5)/0.5)
      loss = criterion(logs, y)

      loss.backward()
      optimizer.step()

      _range.set_description('loss: '+str(loss.item()))

    scheduler.step()

    model.eval()

    valacc = evaluate(args, model, val_loader, len(val))
    print('validation accuracy:', valacc)

    if valacc >= best_valacc:
      best_valacc = valacc
      if not os.path.isdir('./models'):
        os.mkdir('./models')
      ckpt_pth = './models/{}.pth'.format(args.save_name)
      torch.save(model.state_dict(), ckpt_pth)
      print('model saved')

def test(args):

  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img[0, ...].squeeze().unsqueeze(0))])

  test = torchvision.datasets.ImageFolder(
    './data/test'.format(args.frac), transform=transform)

  n_cls=10

  test_loader = torch.utils.data.DataLoader(
    test, 
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True)

  model = ResNet(n_layer=args.n_layer, conv_reduce=args.conv_reduce, n_cls=10).cuda()
  ckpt_pth = './models/{}.pth'.format(args.save_name)
  model.load_state_dict(torch.load(ckpt_pth))
  model.eval()

  acc = evaluate(args, model, test_loader, len(test))

  print('testing accuracy:', acc)

def main(args):

  if args.just_test:
    test(args)
  else:
    train(args)
    test(args)

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--save_name', default='model',
                      help='trained model will get saved under ./models/save_name')
  parser.add_argument('--batch_size', type=int, default=128,
                      help='batch size for training')
  parser.add_argument('--epochs', type=int, default=50,
                      help='number of epochs to train for')
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
  parser.add_argument('--show_data', action='store_true',
                      help='show image data as it is loading')
  parser.add_argument('--just_test', action='store_true',
                      help='if you only want to test and not train')
  args = parser.parse_args()

  main(args)