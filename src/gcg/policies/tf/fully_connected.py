import tensorflow as tf

class FullyConnected(object):
    def __init__(self, num_inputs, num_outputs, activation_fn=None, normalizer_fn=None, normalizer_params=None,
                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                 weights_regularizer=tf.contrib.layers.l2_regularizer(0.5),
                 biases_initializer=tf.constant_initializer(0.),
                 biases_regularizer=None,
                 trainable=True):

        self._W, self._b = self._create_variables(num_inputs,
                                                  num_outputs,
                                                  weights_initializer,
                                                  weights_regularizer,
                                                  biases_initializer,
                                                  biases_regularizer,
                                                  trainable)

        self._activation_fn = activation_fn
        self._normalizer_fn = normalizer_fn
        self._normalizer_params = normalizer_params

    def _create_variables(self, num_inputs, num_outputs, weights_initializer=None, weights_regularizer=None,
                          biases_initializer=None, biases_regularizer=None, trainable=True):

        W = tf.get_variable(
            "W",
            [num_inputs, num_outputs],
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable
        )

        b = None
        if biases_initializer is not None:
            b = tf.get_variable(
                "b",
                [num_outputs],
                initializer=biases_initializer,
                regularizer=biases_regularizer,
                trainable=trainable
            )

        return W, b

    def __call__(self, inputs):
        output = tf.matmul(inputs, self._W)

        if self._normalizer_fn is not None:
            output = self._normalizer_fn(output, **self._normalizer_params)

        if self._b is not None:
            output += self._b

        if self._activation_fn is not None:
            output = self._activation_fn(output)

        return output

