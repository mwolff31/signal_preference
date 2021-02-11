import numpy as np
import imageio

from argparse import ArgumentParser
from tqdm import tqdm
import imutils
import matplotlib.pyplot as plt

from utils import get_pair_matrix, mkdir

def place_img(args, im1, im2, pad):

  # places im2 within im1 such that nonzero pixels do not overlap

  mask = np.copy(im1)
  mask = np.ceil(np.sum(mask, axis=-1))[:,:,None]
  mask = np.minimum(1, mask)

  overlapped = True
  num_tries=0

  while overlapped and num_tries < 10:

    img = np.copy(im1)

    if pad:
      pad = 32*int(args.imgnet_augment)
    else:
      pad = 0

    lu = pad
    rd = args.img_size-args.pattern_size-pad

    if lu==rd:
      xstart = lu
      ystart = lu
    else:
      xstart = np.random.randint(lu, rd)
      ystart = np.random.randint(lu, rd)

    xend = xstart+args.pattern_size
    yend = ystart+args.pattern_size

    add = np.zeros((args.img_size+pad,args.img_size+pad,3))
    add[xstart:xend, ystart:yend, :]+=im2

    img+=add

    overlapped = (img*mask).sum()!=im1.sum()

    num_tries+=1

  img = np.clip(img, 0., 1.)

  if num_tries==10:
    return None
  else:
    return img

def make_img(args, p1, p2, n1, n2, pad=False):

  # places two features together
  # if either p1 or p2 are supposed to have hue pertubations
  # applied, that happens here

  np1 = np.copy(p1)
  np2 = np.copy(p2)

  # hue pertubation
  if n1 in args.color_dev:
    mask = np.float32(np.int32(np1+0.5))
    mask *= np.random.uniform(-args.color_dev_eps, args.color_dev_eps)
    np1 += mask
    np1 = np.clip(np1, 0., 1.)

  if n2 in args.color_dev:
    mask = np.float32(np.int32(np2+0.5))
    mask *= np.random.uniform(-args.color_dev_eps, args.color_dev_eps)
    np2 += mask
    np2 = np.clip(np2, 0., 1.)

  if pad:
    pad = 32*int(args.imgnet_augment)
  else:
    pad = 0

  img = np.zeros((args.img_size+pad, args.img_size+pad, 3))
  img = place_img(args, img, np1, pad)
  img = place_img(args, img, np2, pad)

  # if the overlap check takes >10 tries, p1 gets put
  # in a different spot

  if img is None:
    return make_img(args, p1, p2, n1, n2, pad)
  else:
    return np.uint8(img*255)

def make_set(args, set_id, name_ref, pattern_ref, pair_mat):

  set_cfg = pair_mat[set_id]

  mkdir('./data')
  data_dir = './data/{}'.format(args.exp_name)
  mkdir(data_dir)
  set_dir = './data/{}/set_{}'.format(args.exp_name, set_id)
  mkdir(set_dir)

  num_cls = len(set_cfg)
  splits = ['train', 'val', 'test']

  for split_id in range(len(splits)):
    split = splits[split_id]
    split_dir = '{}/{}'.format(set_dir, split)
    mkdir(split_dir)

    for cls_id in range(num_cls):
      pair_idxs = set_cfg[cls_id]
      cls_name = '{}-{}'.format(name_ref[pair_idxs[0]], name_ref[pair_idxs[1]])
      cls_dir = '{}/{}_{}'.format(split_dir, cls_id, cls_name)
      mkdir(cls_dir)

      for ex_id in range(args.num_ex[split_id]):

        flip = np.random.randint(2)
        id1 = pair_idxs[flip]
        id2 = pair_idxs[1-flip]

        p1 = pattern_ref[id1]
        p2 = pattern_ref[id2]

        if isinstance(p1, list):
          p1 = p1[np.random.randint(len(p1))]
        if isinstance(p2, list):
          p2 = p2[np.random.randint(len(p2))]

        n1 = name_ref[id1]
        n2 = name_ref[id2]

        pad = (split=='train') and args.imgnet_augment

        # predictivity 
        prob = np.random.uniform(0., 1.)
        if n1 in args.pred_drop and (prob >= args.pred_drop_val):
          p1 = np.float32(np.zeros((args.pattern_size,args.pattern_size,3)))

        prob = np.random.uniform(0., 1.)
        if n2 in args.pred_drop and (prob >= args.pred_drop_val):
          p2 = np.float32(np.zeros((args.pattern_size,args.pattern_size,3)))

        img = make_img(args, p1, p2, n1, n2, pad)
        img_name = '{}/{}.png'.format(cls_dir, ex_id)

        imageio.imwrite(img_name, img)

def make_datasets(args, name_ref, pattern_ref):

  pair_mat = get_pair_matrix()

  for set_id in tqdm(range(len(pair_mat))):
    make_set(args, set_id, name_ref, pattern_ref, pair_mat)

def make_cue_conflict(args, name_ref, pattern_ref):

  # render cc_num_ex of each pair in pairs matrix
  # this will be used for all models trained in an experiment

  pair_mat = get_pair_matrix()

  mkdir('./data')
  data_dir = './data/{}'.format(args.exp_name)
  mkdir(data_dir)
  cc_dir = './data/{}/cue_conflict'.format(args.exp_name)

  mkdir(cc_dir)

  split = args.cc_num_ex//2
  num_patterns = len(name_ref)

  for i in tqdm(range(num_patterns)):
    for j in range(num_patterns):
      if j<i:
        namei = name_ref[i]
        namej = name_ref[j]
        ext = '{}-{}'.format(i, j)
        for ex_id in range(args.cc_num_ex):
          if ex_id < split:
            p1 = pattern_ref[i]
            n1 = name_ref[i]
            p2 = pattern_ref[j]
            n2 = name_ref[j]
          else:
            p1 = pattern_ref[j]
            n1 = name_ref[j]
            p2 = pattern_ref[i]
            n2 = name_ref[i]

          if isinstance(p1, list):
            p1 = p1[np.random.randint(len(p1))]
          if isinstance(p2, list):
            p2 = p2[np.random.randint(len(p2))]

          img = make_img(args, p1, p2, n1, n2, False)
          img_name = '{}/{}_{}.png'.format(cc_dir, ext, ex_id)

          imageio.imsave(img_name, img)