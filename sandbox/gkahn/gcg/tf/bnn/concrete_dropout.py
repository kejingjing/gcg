import tensorflow as tf

class ConcreteDropout(object):
    """
    Implements the 'Concrete Dropout' paper by Yarin Gal, Jiri Hron, and Alex Kendall
    https://papers.nips.cc/paper/6949-concrete-dropout
    https://github.com/yaringal/ConcreteDropout
    """

    def __init__(self, layer_name, num_training_data, input_dim, concrete_temperature=0.1, model_precision=1e8,
                 prior_weight_lengthscale=1e-1):
        """

        :param layer_name: the TF name the (logit-ized) 'keep probability' Variable will have
        :param num_training_data: total number of all training data
        :param input_dim: the input dimensionality of the layer this dropout object is being applied to
        :param concrete_temperature: defines how soft/hard the dropout mask is, with lower=harder and higher=softer
        (positive scalar float)
        :param model_precision: inverse likelihood noise variance
        :param prior_weight_lengthscale: standard deviation of the prior weights
        """
        self.layer_name = layer_name
        self.num_training_data = num_training_data
        self.input_dim = input_dim
        self.concrete_temperature = concrete_temperature
        self.model_precision = model_precision # tau in yarin's work
        self.prior_weight_lengthscale = prior_weight_lengthscale  # unsure if this is a good value

        init_keep_prob_logit = tf.get_variable(layer_name + "_logit", initializer=tf.constant(0.))
        self.keep_prob = tf.sigmoid(init_keep_prob_logit)  # map TF variable back to [0,1]
        # self.eps = 1e-7

        self._set_dropout_regulariser()

    def __call__(self, *args, **kwargs):
        return tf

    def _dropout_regulariser_euclidean(self, num_data):
        return 2. / (self.model_precision * num_data)

    def _dropout_regulariser_cross_entropy(self, num_data):
        return 1. / (self.model_precision * num_data)

    @staticmethod
    def _bernoulli_entropy(p):
        entropy = -p * tf.log(p) - (1.-p) * tf.log(1.-p)
        return entropy

    @staticmethod
    def _logit(x):
        # TODO: include eps for numerical stability?
        return tf.log(x) - tf.log(1. - x)

    def _set_dropout_regulariser(self):
        dropout_regularizer = self._dropout_regulariser_cross_entropy(self.num_training_data)
        dropout_regularizer *= self.input_dim * -self._bernoulli_entropy(self.keep_prob)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, dropout_regularizer)

    def get_weight_regularizer_scale(self):
        """Assumes a MSE cost of training data."""
        weight_regularizer = self.prior_weight_lengthscale ** 2 / \
                             (self.model_precision * self.num_training_data) / self.keep_prob
        return weight_regularizer

    def apply_soft_dropout_mask(self, x, uniform_sample):
        # uniform_sample = tf.random_uniform(shape=x.get_shape())
        logit_mask = self._logit(self.keep_prob) + self._logit(uniform_sample)
        soft_dropout_mask = tf.sigmoid(logit_mask / self.concrete_temperature)
        output_layer = x * soft_dropout_mask / self.keep_prob  # rescale
        return output_layer, soft_dropout_mask
