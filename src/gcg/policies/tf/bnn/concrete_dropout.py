import tensorflow as tf

from gcg.policies.tf.fully_connected import FullyConnected

class ConcreteDropout(FullyConnected):
    """
    Implements the 'Concrete Dropout' paper by Yarin Gal, Jiri Hron, and Alex Kendall
    https://papers.nips.cc/paper/6949-concrete-dropout
    https://github.com/yaringal/ConcreteDropout
    """

    def __init__(self,
                 num_data,
                 seed=None,
                 concrete_temperature=0.1,
                 model_precision=1e8,
                 prior_weight_lengthscale=1e-1,
                 create_weights=True,
                 **kwargs):
        self.input_dim = kwargs['num_inputs']
        self.num_data = num_data
        self.seed = seed
        self.concrete_temperature = concrete_temperature
        self.model_precision = model_precision  # tau in yarin's work
        self.prior_weight_lengthscale = prior_weight_lengthscale  # unsure if this is a good value

        init_keep_prob_logit = tf.get_variable("logit", initializer=tf.constant(0.))
        self.keep_prob = tf.sigmoid(init_keep_prob_logit)  # map TF variable back to [0,1]

        self._set_dropout_regulariser()

        kwargs['weights_regularizer'] = tf.contrib.layers.l2_regularizer(self._get_weight_regularizer_scale)

        if create_weights:
            super(ConcreteDropout, self).__init__(**kwargs)

    @property
    def _get_weight_regularizer_scale(self):
        """Assumes a MSE cost of training data."""
        weight_regularizer = self.prior_weight_lengthscale ** 2 / \
                             (self.model_precision * self.num_data) / self.keep_prob
        return weight_regularizer

    def __call__(self, inputs):
        outputs = super(ConcreteDropout, self).__call__(inputs)
        outputs = self.apply_soft_dropout_mask(outputs)
        return outputs

    def apply_soft_dropout_mask(self, x):
        sample = tf.contrib.distributions.Uniform().sample(tf.shape(x), seed=self.seed)
        logit_mask = self._logit(self.keep_prob) + self._logit(sample)
        soft_dropout_mask = tf.sigmoid(logit_mask / self.concrete_temperature)
        output_layer = x * soft_dropout_mask / self.keep_prob  # rescale
        return output_layer

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
        eps = 1e-7  # included for numerical stability
        return tf.log(x + eps) - tf.log(1. - x + eps)

    def _set_dropout_regulariser(self):
        dropout_regularizer = self._dropout_regulariser_cross_entropy(self.num_data)
        dropout_regularizer *= self.input_dim * -self._bernoulli_entropy(self.keep_prob)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, dropout_regularizer)

