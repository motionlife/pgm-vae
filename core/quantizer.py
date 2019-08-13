# Parameter tying with auto-encoder of discrete latent representation
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Re-implement va-vae algor. can perform multiple independent nn calculation"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import initializers as init
from tensorflow.python.training import moving_averages as ma


class VectorQuantizer(Layer):
    """Re-implementation of VQ-VAE https://arxiv.org/abs/1711.00937.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well. (D)
        num_embeddings: integer, the number of vectors in the quantized space. (K)
        commitment_cost: scalar which controls the weighting of the loss terms (beta)
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, num_var, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        super(VectorQuantizer, self).__init__(**kwargs)

    def build(self, input_shape):
        """ The __call__ method of layer will automatically run build the first time it is called.
        Create trainable layer weights (embedding dictionary) here """
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3:
            raise ValueError("The input tensor must be rank of 3")  # (num_fts, batch_size, input_dense_unit)
        num_var = input_shape[0]
        shape = tf.TensorShape([num_var, self.embedding_dim, self.num_embeddings])
        initializer = init.VarianceScaling(distribution='uniform')
        self.embeddings = self.add_weight(name='embeddings', shape=shape, initializer=initializer, trainable=True)
        # Make sure to call the `build` method at the end or set self.built = True
        super(VectorQuantizer, self).build(input_shape)

    def call(self, inputs, training=None, code_only=False, fts=None):
        """ Define the forward computation pass """
        w = self.embeddings if fts is None else tf.gather(self.embeddings, fts, axis=0)
        distances = (tf.reduce_sum(inputs ** 2, 2, keepdims=True)
                     - 2 * tf.matmul(inputs, w)
                     + tf.reduce_sum(w ** 2, 1, keepdims=True))
        encoding_indices = tf.argmin(distances, 2)
        if not code_only:
            quantized = tf.gather(tf.transpose(w, [0, 2, 1]), encoding_indices, axis=1, batch_dims=1)
            e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)  # commitment loss
            q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            output = inputs + tf.stop_gradient(quantized - inputs)  # grad(Zq(x)) = grad(Ze(x))
        else:
            loss = 0.
            output = tf.one_hot(encoding_indices, self.num_embeddings) if fts is None else encoding_indices

        self.add_loss(loss)
        return output

    def get_config(self):
        base_config = super(VectorQuantizer, self).get_config()
        base_config.update({'_embedding_dim': self.embedding_dim,
                            '_num_embeddings': self.num_embeddings,
                            '_commitment_cost': self.commitment_cost
                            })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VectorQuantizerEMA(Layer):
    """ Implements a slightly modified version of the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    The difference between VectorQuantizerEMA and VectorQuantizer is that
    this module uses exponential moving averages to update the embedding vectors
    instead of an auxiliary loss. This has the advantage that the embedding
    updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
    ...) used for the encoder, decoder and other parts of the architecture. For
    most experiments the EMA version trains faster than the non-EMA version.

    Args:
      embedding_dim: integer representing the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
      num_embeddings: integer, the number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms (see
        equation 4 in the paper).
      decay: float, decay for the moving averages.
      epsilon: small float constant to avoid numerical instability.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay, num_var,
                 epsilon=1e-5, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        super(VectorQuantizerEMA, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3:
            raise ValueError("Input shape must be 3")
        # last_dim = input_shape[-1]
        num_var = input_shape[0]
        shape = tf.TensorShape([num_var, self.embedding_dim, self.num_embeddings])
        initializer = init.VarianceScaling(distribution='uniform')
        self.embeddings = self.add_weight(name='embeddings', shape=shape, initializer=initializer)
        self.ema_cluster_size = self.add_weight(name='ema_cluster_size', shape=[num_var, self.num_embeddings],
                                                initializer=init.get('zeros'))
        self.ema_w = self.add_weight(name='ema_dw', shape=shape)
        self.ema_w.assign(self.embeddings.read_value())
        super(VectorQuantizerEMA, self).build(input_shape)

    def call(self, inputs, training=None, code_only=False, fts=None):
        """forward pass computation
        Args:
          inputs: Tensor, final dimension must be equal to embedding_dim. All other
            leading dimensions will be kept as-is and treated as a large batch.
          training: boolean, whether this connection is to training data. When
            this is set to False, the internal moving average statistics will not be
            updated.
         code_only: if only return encoding index
         fts: indices of features, none means all of them

        Returns:
          quantized tensor which has the same shape as input tensor.
        """
        w = self.embeddings if fts is None else tf.gather(self.embeddings, fts, axis=0)
        distances = (tf.reduce_sum(inputs ** 2, 2, keepdims=True)
                     - 2 * tf.matmul(inputs, w)
                     + tf.reduce_sum(w ** 2, 1, keepdims=True))
        encoding_indices = tf.argmin(distances, 2)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        if not code_only:
            quantized = tf.gather(tf.transpose(w, [0, 2, 1]), encoding_indices, axis=1, batch_dims=1)
            e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
            if training:
                updated_ema_cluster_size = ma.assign_moving_average(
                    self.ema_cluster_size, tf.reduce_sum(encodings, 1), self.decay)
                dw = tf.matmul(inputs, encodings, transpose_a=True)
                updated_ema_w = ma.assign_moving_average(self.ema_w, dw, self.decay)
                n = tf.reduce_sum(updated_ema_cluster_size, axis=1, keepdims=True)
                updated_ema_cluster_size = (updated_ema_cluster_size + self.epsilon) / (
                        n + self.num_embeddings * self.epsilon) * n
                normalised_updated_ema_w = updated_ema_w / tf.expand_dims(updated_ema_cluster_size, 1)
                w.assign(normalised_updated_ema_w)
                loss = self.commitment_cost * e_latent_loss
            else:
                loss = self.commitment_cost * e_latent_loss
            output = inputs + tf.stop_gradient(quantized - inputs)
        else:
            loss = 0.
            output = encodings if fts is None else encoding_indices

        self.add_loss(loss)
        return output

    def get_config(self):
        base_config = super(VectorQuantizerEMA, self).get_config()
        base_config.update({'_embedding_dim': self.embedding_dim,
                            '_num_embeddings': self.num_embeddings,
                            '_commitment_cost': self.commitment_cost,
                            '_decay': self.decay,
                            '_epsilon': self.epsilon
                            })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class VectorQuantizerNaive(Layer):
    def __init__(self, dim, commitment_cost, **kwargs):
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.num_embeddings = 2 ** dim
        self.power = 2 ** tf.range(dim)
        super(VectorQuantizerNaive, self).__init__(**kwargs)

    def call(self, inputs, training=None, code_only=False, fts=None):
        z = inputs  # 1. / (1 + tf.exp(-27 * (inputs - 0.5))) # todo: a continuous round function
        # quantized = tf.minimum(tf.maximum(z - 0.499999, 0) * 1e7, 1)
        if not code_only:
            # todo: penalize z concentrate new 0, encourage them spread on two ends (-inf or inf)
            # loss = -tf.reduce_sum(z * tf.math.log(z + 1e-10)) * self.commitment_cost ???
            loss = self.commitment_cost * tf.reduce_mean(-(z - 0.5) ** 2)
            output = tf.minimum(tf.maximum(z - 0.499999, 0) * 1e7, 1)
        else:
            loss = 0.
            enc_idx = tf.cast(tf.reduce_sum(tf.cast(tf.round(z), tf.int32) * self.power, axis=-1), tf.int64)
            output = tf.one_hot(enc_idx, self.num_embeddings) if fts is None else enc_idx

        self.add_loss(loss)
        return output
