import argparse, os, glob

import numpy as np
from PIL import Image
import scipy.misc

import tensorflow as tf

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

### setup tensorflow writing
writer = tf.python_io.TFRecordWriter(os.path.join(folder, 'data.tfrecords'))

### read image, label pairs
label_fnames = glob.glob(os.path.join(folder, 'label*'))
images = []
labels = []
for label_fname in label_fnames:
    image_fname = label_fname.replace('label_', '')

    image = np.asarray(Image.open(image_fname))
    label = np.asarray(Image.open(label_fname))

    # reduce image back down
    image = image[border:-border, border:-border]
    image = scipy.misc.imresize(image, 1. / float(resize), interp='lanczos')
    assert(tuple(image.shape) == reshape)

    # reduce label back down
    label = label[border:-border, border:-border]
    label = scipy.misc.imresize(label.astype(np.float32), 1. / float(resize), interp='bilinear')
    label = (label > 0.5)
    assert(tuple(label.shape) == reshape)

    # tfrecords

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    height, width = image.shape
    image_raw = image.tostring()
    label_raw = label.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(image_raw),
        'label_raw': _bytes_feature(label_raw)
    }))
    writer.write(example.SerializeToString())

    images.append(image)
    labels.append(label)

np.save(os.path.join(folder, 'data_images.npy'), np.array(images))
np.save(os.path.join(folder, 'data_labels.npy'), np.array(labels))


# import matplotlib.pyplot as plt
# im = 255 * np.ones(reshape, np.uint8) * label
# plt.imshow(im, cmap='Greys_r')
# # plt.imshow(image, cmap='Greys_r')
# plt.show(block=False)
# plt.pause(0.1)
# import IPython; IPython.embed()