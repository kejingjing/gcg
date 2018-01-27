import tensorflow as tf

class Bootstrap(object):

    def __init__(self, num_bootstraps):
        self._num_bootstraps = num_bootstraps

    def __call__(self, inputs, num_outputs, activation_fn=None, normalizer_fn=None, normalizer_params=None,
                 weights_initializer=None, weights_regularizer=None, biases_initializer=None, biases_regularizer=None,
                 trainable=True):
        """
        Assumes inputs are [num_bootstraps * batch_size, -1]
        aka the data for each bootstrap is contiguous along the first dimension
        """
        outputs = []
        for b, inputs_b in enumerate(tf.split(inputs, self._num_bootstraps, axis=0)):
            with tf.variable_scope('bootstrap_{0}'.format(b)):
                outputs_b = tf.contrib.layers.fully_connected(
                    inputs=inputs_b,
                    num_outputs=num_outputs,
                    activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params,
                    weights_initializer=weights_initializer,
                    biases_initializer=biases_initializer,
                    weights_regularizer=weights_regularizer,
                    trainable=trainable)
                outputs.append(outputs_b)

        return tf.concat(outputs, 0)