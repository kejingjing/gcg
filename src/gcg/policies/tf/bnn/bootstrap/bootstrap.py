import tensorflow as tf

from gcg.policies.tf.fully_connected import FullyConnected

class Bootstrap(object):
    def __init__(self, num_bootstraps, **kwargs):
        self._num_bootstraps = num_bootstraps
        self._fcs = []
        for b in range(num_bootstraps):
            with tf.variable_scope('bootstrap{0}'.format(b)):
                self._fcs.append(FullyConnected(**kwargs))

    def __call__(self, inputs):
        outputs = []
        for fc_b, inputs_b in zip(self._fcs, tf.split(inputs, self._num_bootstraps)):
            outputs.append(fc_b(inputs_b))

        return tf.concat(outputs, 0)
