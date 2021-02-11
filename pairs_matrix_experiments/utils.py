import os
import numpy as np
import imageio
import glob
import cv2

def resize(image, s, inter=cv2.INTER_AREA):

    resized = cv2.resize(image, (s,s), interpolation=inter)

    return resized

def get_pair_matrix():

  # pair matrix definition

  s1 = [[0,1],[2,3],[4,6],[5,7],[8,9]]
  s2 = [[0,2],[1,5],[3,4],[6,8],[7,9]]
  s3 = [[0,3],[1,2],[4,5],[7,8],[6,9]]
  s4 = [[0,4],[1,6],[3,7],[2,8],[5,9]]
  s5 = [[0,5],[1,7],[2,6],[3,8],[4,9]]
  s6 = [[0,6],[1,4],[2,7],[5,8],[3,9]]
  s7 = [[0,7],[1,3],[5,6],[4,8],[2,9]]
  s8 = [[0,8],[4,7],[3,6],[2,5],[1,9]]
  s9 = [[0,9],[1,8],[6,7],[3,5],[2,4]]

  pair_idxs = [s1,s2,s3,s4,s5,s6,s7,s8,s9]

  return pair_idxs

def mkdir(d):

  if not os.path.isdir(d):
    os.mkdir(d)

def read_pattern(args, pth):

  # read a feature image (pattern)

  img = imageio.imread(pth)
  if len(img.shape) < 3:
    img = np.tile(img[:,:,None], (1,1,3))
  if img.shape[-1] > 3:
    img = img[:, :, :3]
  if img.shape[0]!=args.pattern_size or img.shape[1]!=args.pattern_size:
    img = resize(img, args.pattern_size)
  if img.max() > 1:
    img = img/255
    
  return img

def read_patterns(args):

  # create a dictionary reference for features with indices
  # from pairs matrix
  # also creates a reference for featue names

  pths = glob.glob('./patterns/{}/*'.format(args.pattern_dir))
  pths = sorted(pths)

  patterns = []
  names = []

  for pth in pths:
    if os.path.isdir(pth):
      pths2 = glob.glob(pth+'/*')
      inner = []
      for pth2 in pths2:
        inner.append(read_pattern(args, pth2))
      patterns.append(inner)
    else:
      patterns.append(read_pattern(args, pth))
    names.append(pth)

  names = [name.split('/')[-1].split('.')[0] for name in names]

  r = list(range(len(names)))
  name_ref = dict(zip(r,names))
  pattern_ref = dict(zip(r,patterns))

  return name_ref, pattern_ref

def quantize(x):

  x = np.sum(x, axis=-1)
  x = np.minimum(1, x)
  x = np.int32(x+0.5)

  return x

def count_pixels(x):

  # count the number of pixels used to represent the pattern

  return quantize(x).sum()

def get_pixel_ref(pattern_ref):

  # get pixel dictionary
  # reference w/ indices for pairs matrix

  keys = list(pattern_ref.keys())
  pixel_ref = {}

  for key in keys:

    pattern = pattern_ref[key]
    if isinstance(pattern, list):
      total = 0
      for p in pattern:
        total+=count_pixels(p)
      total = total / len(pattern)
    else:
      total=count_pixels(pattern)

    pixel_ref[key] = total

  return pixel_ref