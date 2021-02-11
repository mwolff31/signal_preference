import numpy as np
import os
import torch
import imageio

def make_splits(dataset_size, frac):

  # create and save train/val splits

  if not os.path.exists('./splits/train_idxs_100.npy') and frac==1:
    if not os.path.isdir('./splits'):
      os.mkdir('./splits')
    idxs = np.arange(dataset_size)
    np.random.shuffle(idxs)
    train_idxs = idxs[:55000]
    val_idxs = idxs[55000:]
    np.save('./splits/train_idxs_100.npy', train_idxs)
    np.save('./splits/val_idxs.npy', val_idxs)

  else:
    train_idxs = np.load('./splits/train_idxs_{}.npy'.format(int(frac*100)))
    val_idxs = np.load('./splits/val_idxs.npy')

  return train_idxs, val_idxs

def acc(pred, actual):

  # accuracy helper

  return np.int32(pred==actual).sum() / actual.size

def get_img(x):

  # torch tensor --> numpy image

  return x.permute(1,2,0).cpu().detach().numpy()

def mkdir(d):

  # mkdir helper

  if not os.path.isdir(d):
    os.mkdir(d)

def corr(x, y):

  # Pearson correlation coefficient

  xc = x - np.mean(x)
  yc = y - np.mean(y)

  cov = np.mean(xc*yc)

  norm = np.std(xc)*np.std(yc)

  return cov / norm