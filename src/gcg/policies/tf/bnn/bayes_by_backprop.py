import tensorflow as tf

class BayesByBackprop(object):
    """
    Implements the 'Weight Uncertainty in Neural Networks' paper Charles Blundell et al.
    http://proceedings.mlr.press/v37/blundell15.pdf
    https://arxiv.org/abs/1505.05424
    """

    def __init__(self, layer_name, num_data, batch_size):
        self.layer_name = layer_name
        self.num_data = num_data
        self.batch_size = batch_size

    @staticmethod
    def _make_positive(x):
            return tf.nn.softplus(x)

    def _get_weights_or_biases_and_regularize(self, batch_size, shape, mu_initializer, regularizer, str_weights_or_biases):
        tensor_name = "{}-{}".format(self.layer_name, str_weights_or_biases)
        mu = tf.get_variable(tensor_name + '_mu', shape=shape, initializer=mu_initializer)
        rho = tf.get_variable(tensor_name + '_rho', initializer=tf.constant(0.1, shape=shape))
        sigma = BayesByBackprop._make_positive(rho)
        noise_sample = tf.random_normal([batch_size] + shape)

        weights_or_biases = mu + sigma * noise_sample

        kl_regularization = BayesByBackprop._get_kl_regularization(mu, sigma)
        kl_regularization *= tf.to_float(self.batch_size) / self.num_data  # minibatch scaling
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, kl_regularization)
        # tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer(weight_or_bias))

        return weights_or_biases

    @staticmethod
    def _get_kl_regularization(mean_q, sigma_q, mean_p=0.0, sigma_p=1.0):
        var_q = tf.square(sigma_q)
        var_p = tf.square(sigma_p)
        kl = (tf.square(mean_q - mean_p) + var_q) / (2.*var_p) + 0.5*(tf.log(var_p) - tf.log(var_q) - 1.)
        return tf.reduce_sum(kl)

    def get_weight_regularizer_scale(self):
        return 0.5

    # Note: should only be called once per instance
    def __call__(self, inputs, num_outputs, activation_fn=None, normalizer_fn=None, normalizer_params=None,
                 weights_initializer=None, weights_regularizer=None, biases_initializer=None, biases_regularizer=None,
                 trainable=True):
        # TODO: sort out normalisers...I don't use them yet.
        batch_size = tf.shape(inputs)[0]

        # weights, shape: [batch_size, dim_input, dim_output]
        weights_shape = [inputs.get_shape()[1].value, num_outputs]
        if weights_initializer is None:
            weights_initializer = tf.truncated_normal(weights_shape, stddev=0.01)
        weights = self._get_weights_or_biases_and_regularize(batch_size, weights_shape, weights_initializer, weights_regularizer,
                                                             'weights')

        # biases, shape: [batch_size, dim_outputs]
        biases_shape = [num_outputs]
        if biases_regularizer is None:
            biases_regularizer = weights_regularizer
            # biases_initializer = tf.constant(0.0, shape=biases_shape)
        biases = self._get_weights_or_biases_and_regularize(batch_size, biases_shape, biases_initializer, biases_regularizer,
                                                            'biases')

        # the layer, inputs shape: [?, dim_inputs]
        fc_layer_out = tf.reduce_sum(inputs[:,:,None] * weights, axis=1) + biases  # shape [?, num_outputs]
        if activation_fn is not None:
            fc_layer_out = activation_fn(fc_layer_out)

        return fc_layer_out
