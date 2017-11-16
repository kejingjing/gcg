import tensorflow as tf

TF_VERSION = float('.'.join(tf.__version__.split('.')[:-1]))
assert(TF_VERSION >= 1)

def global_variables_initializer():
    return tf.global_variables_initializer()

def global_variables_collection_name():
    return tf.GraphKeys.GLOBAL_VARIABLES

def trainable_variables_collection_name():
    return tf.GraphKeys.TRAINABLE_VARIABLES

def variables_initializer(vars):
    return tf.variables_initializer(vars)

def split(value, num_splits, axis=0):
    return tf.split(value, num_splits, axis=axis)

def concat(values, axis):
    return tf.concat(values, axis)
