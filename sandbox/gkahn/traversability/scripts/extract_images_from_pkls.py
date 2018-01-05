import argparse, os, glob

import numpy as np
from PIL import Image

from sandbox.gkahn.gcg.utils import mypickle

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str)
parser.add_argument('-reshape', type=str, default='(36, 64)')
parser.add_argument('-subsample', type=int, default=4)
parser.add_argument('-border', type=int, default=10)
args = parser.parse_args()

folder = args.folder
reshape = eval(args.reshape)
subsample = args.subsample
border = args.border

print('')
print('folder: {0}'.format(folder))
print('reshape: {0}'.format(reshape))
print('subsample: {0}'.format(subsample))
print('border: {0}'.format(border))
print('')

### convert
fnames = sorted(glob.glob(os.path.join(folder, '*_train_rollouts.pkl')))
print('{0} files to read'.format(len(fnames)))
image_folder = os.path.join(folder, 'images')
os.makedirs(image_folder, exist_ok=True)

im_num = 0
for fname in fnames:
    rollouts = mypickle.load(fname)['rollouts']
    for rollout in rollouts:
        for obs in rollout['observations'][::subsample]:
            im = np.reshape(obs, reshape)
            im = np.pad(im, ((border, border), (border, border)), 'constant')
            Image.fromarray(im).save(os.path.join(image_folder, 'image_{0:06d}.jpg'.format(im_num)))
            im_num += 1

print('Saved {0} images'.format(im_num))
print('')
