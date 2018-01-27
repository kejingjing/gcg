import tensorflow as tf

from gcg.policies.tf.fully_connected import FullyConnected

class BayesByBackprop(FullyConnected):
    """
    Implements the 'Weight Uncertainty in Neural Networks' paper Charles Blundell et al.
    http://proceedings.mlr.press/v37/blundell15.pdf
    https://arxiv.org/abs/1505.05424
    """
    def __init__(self, num_data, batch_size, **kwargs):
        self.num_data = num_data
        self.batch_size = batch_size

        super(BayesByBackprop, self).__init__(**kwargs)

    def _create_variables(self, num_inputs, num_outputs, weights_initializer=None, weights_regularizer=None,
                          biases_initializer=None, biases_regularizer=None, trainable=True):
        with tf.variable_scope('W'):
            W = self._create_bbb_variable([num_inputs, num_outputs], weights_initializer, weights_regularizer,
                                          trainable=trainable)

        b = None
        if biases_initializer is not None:
            if biases_regularizer is None:
                biases_regularizer = weights_regularizer

            with tf.variable_scope('b'):
                b = self._create_bbb_variable([num_outputs], biases_initializer, biases_regularizer,
                                              trainable=trainable)

        return W, b

    def _create_bbb_variable(self, shape, mu_initializer, regularizer, trainable):
        mu = tf.get_variable(
            'mu',
            shape,
            initializer=mu_initializer,
            trainable=trainable
        )
        rho = tf.get_variable(
            'rho',
            initializer=tf.constant(0.1, shape=shape)
        )
        sigma = tf.nn.softplus(rho) # make positive
        noise_sample = tf.random_normal(shape)

        var = mu + sigma * noise_sample

        kl_regularization = BayesByBackprop._get_kl_regularization(mu, sigma)
        kl_regularization *= tf.to_float(self.batch_size) / self.num_data  # minibatch scaling
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, kl_regularization)

        return var

    @staticmethod
    def _get_kl_regularization(mean_q, sigma_q, mean_p=0.0, sigma_p=1.0):
        var_q = tf.square(sigma_q)
        var_p = tf.square(sigma_p)
        kl = (tf.square(mean_q - mean_p) + var_q) / (2.*var_p) + 0.5*(tf.log(var_p) - tf.log(var_q) - 1.)
        return tf.reduce_sum(kl)
