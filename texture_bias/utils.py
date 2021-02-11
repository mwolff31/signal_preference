import cv2
import csv
import numpy as np
import glob
import imageio

def resize(image, s, inter=cv2.INTER_AREA):

  # resize an image
  # used from the imutils pacakge

  resized = cv2.resize(image, (s,s), interpolation=inter)

  return resized

class Stimulus(object):

  def __init__(self, path, args):

    self.path = path
    self.args = args
    self.category = self.path.split('/')[-2]
    self.name = self.path.split('/')[-1].split('.')[0]

    self.shape_class = 0
    self.texture_class = 0
    self.valid_occur = 0
    self.occur = 0

    pth = './stimuli/filled-silhouettes/'

    sil_path = pth+'{}/{}.png'.format(self.category, self.name)

    sil_img = 1. - imageio.imread(sil_path)/255.0

    # resize image and sil
    assert args.size <= 1
    if args.size < 1:
      s = int(224*args.size)
      sil = resize(sil_img, s)
      pad1 = (224-s)//2
      pad2 = pad1+(224-s)%2
      sil = np.pad(sil, ((pad1, pad2), (pad1, pad2), (0,0)), 
        'constant', constant_values=0)

    else:
      sil = sil_img

    self.mask = sil_img

def get_stim_ref(args):

  # return dictionary of Stimulus objects for each
  # silhouette name (160 total)

  cats = glob.glob('./stimuli/edges/*')
  total = []
  for cat in cats:
    inds = glob.glob(cat+'/*')
    total+=inds

  stims = []
  names = []
  for i in range(len(total)):
    path = total[i]
    stim = Stimulus(path, args)
    stims.append(stim)
    names.append(stim.name)

  stim_ref = dict(zip(names,stims))

  return stim_ref

def get_scapes():

  # read landscape images

  scape_pths = glob.glob('./landscapes/*')
  scapes = []
  for pth in scape_pths:
    scape = imageio.imread(pth)/255.0
    if scape.shape[-1] > 3:
      scape = scape[:, :, :3]
    scape = resize(scape, 224)
    scapes.append(scape)

  return scapes