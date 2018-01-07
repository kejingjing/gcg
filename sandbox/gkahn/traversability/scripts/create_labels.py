import argparse, os

import pandas
import numpy as np
from matplotlib.path import Path
from PIL import Image

### argparse
parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str)
parser.add_argument('-reshape', type=str, default='(36, 64)')
parser.add_argument('-resize', type=float, default=10.)
parser.add_argument('-border', type=int, default=100)
args = parser.parse_args()

folder = args.folder
reshape = eval(args.reshape)
resize = args.resize
border = args.border

print('')
print('folder: {0}'.format(folder))
print('reshape: {0}'.format(reshape))
print('resize: {0}'.format(resize))
print('border: {0}'.format(border))
print('')

### load csv
csv_fname = os.path.join(folder, 'via_region_data.csv')
assert(os.path.exists(csv_fname))
csv = pandas.read_csv(csv_fname)

### image indices
xdim = int(resize)*reshape[0] + 2*border
ydim = int(resize)*reshape[1] + 2*border
x, y = np.meshgrid(np.arange(xdim), np.arange(ydim))
indices = np.vstack((y.flatten(), x.flatten())).T

### create labels
num_labelled = 0
for i in range(len(csv)):
    fname = csv['#filename'][i]
    region_shape_attrs = eval(csv['region_shape_attributes'][i])
    if 'name' not in region_shape_attrs.keys():
        continue

    assert(region_shape_attrs['name'] == 'polygon')
    xy = np.stack((region_shape_attrs['all_points_x'], region_shape_attrs['all_points_y'])).T
    label = 1 - Path(xy).contains_points(indices).reshape(ydim, xdim).T # 0 is no collision
    label = label.astype(np.uint8)

    label_fname = os.path.join(folder, 'label_' + os.path.splitext(fname)[0] + '.jpg')
    Image.fromarray(label).save(label_fname)

    num_labelled += 1

print('{0} were labelled'.format(num_labelled))
print('')

import matplotlib.pyplot as plt
im = 255 * np.ones((xdim, ydim), np.uint8) * label
plt.imshow(im, cmap='Greys_r')
plt.show(block=False)
plt.pause(0.1)

import IPython; IPython.embed()