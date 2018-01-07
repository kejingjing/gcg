import argparse, os, glob
import random, itertools

import numpy as np
from PIL import Image
import scipy.misc

from sandbox.gkahn.gcg.utils import mypickle

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str)
parser.add_argument('-maxsaved', type=int, default=1000)
parser.add_argument('-reshape', type=str, default='(36, 64)')
parser.add_argument('-resize', type=float, default=10.)
parser.add_argument('-border', type=int, default=100)
args = parser.parse_args()

folder = args.folder
maxsaved = args.maxsaved
reshape = eval(args.reshape)
resize = args.resize
border = args.border

print('')
print('folder: {0}'.format(folder))
print('maxsaved: {0}'.format(maxsaved))
print('reshape: {0}'.format(reshape))
print('resize: {0}'.format(resize))
print('border: {0}'.format(border))
print('')

### convert
fnames = glob.glob(os.path.join(folder, '*_train_rollouts.pkl'))
random.shuffle(fnames)
print('{0} files to read'.format(len(fnames)))
fnames = itertools.cycle(fnames)
image_folder = os.path.join(folder, 'images')
os.makedirs(image_folder, exist_ok=True)

im_num = 0
while im_num < maxsaved:
    fname = next(fnames)
    rollout = random.choice(mypickle.load(fname)['rollouts'])
    obs = random.choice(rollout['observations'])

    im = np.reshape(obs, reshape)
    im = scipy.misc.imresize(im, resize, interp ='lanczos')
    im = np.pad(im, ((border, border), (border, border)), 'constant')
    Image.fromarray(im).save(os.path.join(image_folder, 'image_{0:06d}.jpg'.format(im_num)))
    im_num += 1

print('Saved {0} images'.format(im_num))
print('')
