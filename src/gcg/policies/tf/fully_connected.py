import tensorflow as tf

class FullyConnected(object):
    def __init__(self, num_inputs, num_outputs, activation_fn=None, normalizer_fn=None, normalizer_params=None,
                 weights_initializer=None, weights_regularizer=None, biases_initializer=None, biases_regularizer=None,
                 trainable=True):
        self._W = tf.get_variable(
                "W",
                [num_inputs, num_outputs],
                initializer=weights_initializer,
                regularizer=weights_regularizer,
                trainable=trainable
            )

        self._b = None
        if biases_initializer is not None:
            self._b = tf.get_variable(
                "b",
                [num_outputs],
                initializer=biases_initializer,
                regularizer=biases_regularizer,
                trainable=trainable
            )

        self._activation_fn = activation_fn
        self._normalizer_fn = normalizer_fn
        self._normalizer_params = normalizer_params

    def __call__(self, inputs):
        output = tf.matmul(inputs, self._W)

        if self._normalizer_fn is not None:
            output = self._normalizer_fn(output, **self._normalizer_params)

        if self._b is not None:
            output += self._b

        if self._activation_fn is not None:
            output = self._activation_fn(output)

        return output

