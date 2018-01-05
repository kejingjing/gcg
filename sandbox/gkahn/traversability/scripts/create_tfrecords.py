import argparse, os, glob

import numpy as np
from PIL import Image

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str)
parser.add_argument('-reshape', type=str, default='(36, 64)')
parser.add_argument('-border', type=int, default=10)
args = parser.parse_args()

folder = args.folder
reshape = eval(args.reshape)
border = args.border

print('')
print('folder: {0}'.format(folder))
print('reshape: {0}'.format(reshape))
print('border: {0}'.format(border))
print('')

### read image, label pairs
label_fnames = glob.glob(os.path.join(folder, 'label*'))
for label_fname in label_fnames:
    image_fname = label_fname.replace('label_', '')

    image = np.asarray(Image.open(image_fname))
    label = np.asarray(Image.open(label_fname))

import IPython; IPython.embed()