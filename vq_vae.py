# Parameter tying with auto-encoder of discrete latent representation
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Customized tensorflow keras model with customized vector quantization layer"""

import os
import time
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from parallel_dense import ParallelDense
from tensorflow.python.training import moving_averages


class VectorQuantizer(Layer):
    """Re-implementation of VQ-VAE https://arxiv.org/abs/1711.00937.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well. (D)
        num_embeddings: integer, the number of vectors in the quantized space. (K)
        commitment_cost: scalar which controls the weighting of the loss terms (beta)
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, **kwargs):
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        super(VectorQuantizer, self).__init__(**kwargs)

    def build(self, input_shape):
        """ The __call__ method of layer will automatically run build the first time it is called.
        Create trainable layer weights (embedding dictionary) here """
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3:
            raise ValueError("The input tensor must be rank of 3")  # (num_fts, batch_size, input_dense_unit)
        last_dim = input_shape[-1]
        num_fts = input_shape[0]
        with tf.control_dependencies([tf.Assert(tf.equal(last_dim, self._embedding_dim), [last_dim])]):
            shape = tf.TensorShape([num_fts, self._embedding_dim, self._num_embeddings])
        initializer = tf.keras.initializers.he_uniform(seed=7)  # GlorotUniform(seed=7)?
        self._w = self.add_weight(name='embeddings',
                                         shape=shape,
                                         initializer=initializer,
                                         trainable=True)
        # Make sure to call the `build` method at the end or set self.built = True
        super(VectorQuantizer, self).build(input_shape)

    def call(self, inputs):
        """ Define the forward computation pass """
        distances = (tf.reduce_sum(inputs ** 2, 2, keepdims=True)
                     - 2 * tf.matmul(inputs, self._w)
                     + tf.reduce_sum(self._w ** 2, 1, keepdims=True))

        enc_idx = tf.argmin(distances, 2)
        with tf.control_dependencies([enc_idx]):  # batch gather
            quantized = tf.gather(tf.transpose(self._w, [0, 2, 1]), enc_idx, axis=1, batch_dims=1)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)  # commitment loss
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        self.add_loss(loss)

        quantized = inputs + tf.stop_gradient(quantized - inputs)  # grad(Zq(x)) = grad(Ze(x))
        return quantized
        # encodings = tf.one_hot(enc_idx, self._num_embeddings)
        # avg_probs = tf.reduce_mean(encodings, 0)
        # perplexity = tf.math.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10))) # perplexity = exp(H(p))
        # self._pkgs = {'quantized': quantized,
        #               'loss': loss,
        #               'perplexity': perplexity,
        #               'encodings': encodings,
        #               'enc_idx': enc_idx, }

    @property
    def embeddings(self):
        return self._w

    @staticmethod
    def compute_output_shape(input_shape):
        # return tf.TensorShape(input_shape)
        return input_shape

    # Optionally, a layer can be serialized by implementing the get_config method and the from_config class method.
    def get_config(self):
        base_config = super(VectorQuantizer, self).get_config()
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
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
                 epsilon=1e-5, **kwargs):
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._decay = decay
        self._commitment_cost = commitment_cost
        self._epsilon = epsilon
        super(VectorQuantizerEMA, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3:
            raise ValueError("Input shape must be 3")
        last_dim = input_shape[-1]
        num_fts = input_shape[0]
        with tf.control_dependencies([tf.Assert(tf.equal(last_dim, self._embedding_dim), [last_dim])]):
            shape = tf.TensorShape([num_fts, self._embedding_dim, self._num_embeddings])
        initializer = tf.keras.initializers.he_uniform(seed=7)  # GlorotUniform(seed=7)?
        # w is a matrix with an embedding in each column. When training, the embedding
        # is assigned to be the average of all inputs assigned to that  embedding.
        self._w = self.add_weight(name='embeddings', shape=shape,
                                  initializer=initializer, use_resource=True)
        self._ema_cluster_size = self.add_weight(name='ema_cluster_size', shape=[num_fts, self._num_embeddings],
                                                 initializer=tf.constant_initializer(0), use_resource=True)
        self._ema_w = self.add_weight(name='ema_dw', shape=shape, use_resource=True)
        self._ema_w.assign(self._w.read_value())
        super(VectorQuantizerEMA, self).build(input_shape)

    def call(self, inputs, training=None):
        """forward pass computation
        Args:
          inputs: Tensor, final dimension must be equal to embedding_dim. All other
            leading dimensions will be kept as-is and treated as a large batch.
          training: boolean, whether this connection is to training data. When
            this is set to False, the internal moving average statistics will not be
            updated.

        Returns:
          quantized tensor which has the same shape as input tensor.
        """
        with tf.control_dependencies([inputs]):
            w = self._w.read_value()

        distances = (tf.reduce_sum(inputs ** 2, 2, keepdims=True)
                     - 2 * tf.matmul(inputs, w)
                     + tf.reduce_sum(w ** 2, 1, keepdims=True))
        enc_idx = tf.argmin(distances, 2)
        encodings = tf.one_hot(enc_idx, self._num_embeddings)
        quantized = tf.gather(tf.transpose(w, [0, 2, 1]), enc_idx, axis=1, batch_dims=1)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)

        if training:
            updated_ema_cluster_size = moving_averages.assign_moving_average(
                self._ema_cluster_size, tf.reduce_sum(encodings, 1), self._decay)
            dw = tf.matmul(inputs, encodings, transpose_a=True)
            updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw, self._decay)
            n = tf.reduce_sum(updated_ema_cluster_size, axis=1, keepdims=True)
            updated_ema_cluster_size = (updated_ema_cluster_size + self._epsilon) / (
                    n + self._num_embeddings * self._epsilon) * n
            normalised_updated_ema_w = (
                    updated_ema_w / tf.expand_dims(updated_ema_cluster_size, 1))
            with tf.control_dependencies([e_latent_loss]):
                update_w = self._w.assign(normalised_updated_ema_w)
                with tf.control_dependencies([update_w]):
                    loss = self._commitment_cost * e_latent_loss
        else:
            loss = self._commitment_cost * e_latent_loss

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        self.add_loss(loss)
        return quantized

    @property
    def embeddings(self):
        return self._w

    @staticmethod
    def compute_output_shape(input_shape):
        return input_shape


class ParVAE(Model):
    """A customized model derived from Keras model for batch features training"""

    def __init__(self, fts=15, emb=30, dim=8, cost=0.25):
        super(ParVAE, self).__init__(name='parallel_vae')
        # may try dropout layer to do regularization
        self.dense_1 = ParallelDense(12, activation='relu')
        self.dense_2 = ParallelDense(10, activation='relu')
        self.dense_3 = ParallelDense(dim, activation='relu')
        # self.vq_layer = VectorQuantizer(embedding_dim=dim, num_embeddings=emb, commitment_cost=cost)
        self.vq_layer = VectorQuantizerEMA(embedding_dim=dim, num_embeddings=emb, commitment_cost=cost, decay=0.99)
        self.dense_4 = ParallelDense(10, activation='relu')
        self.dense_5 = ParallelDense(12, activation='relu')
        self.dense_6 = ParallelDense(fts, activation='sigmoid')  # make sure the output of the model is [0,1]

    def call(self, inputs):
        x = tf.transpose(inputs, [1, 0, 2])
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.vq_layer(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)
        return tf.transpose(x, [1, 0, 2])

    @staticmethod
    def compute_output_shape(input_shape):
        return input_shape


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # using cpu
    log_dir = os.path.join(os.path.join(os.curdir, "logs"), time.strftime("run_%Y_%m_%d-%H_%M_%S"))

    # test layer
    batch = 100
    D = 8
    K = 50
    beta = 0.2

    train_ds = tf.data.TextLineDataset('trw/nltcs.ts.data') \
        .map(lambda x: tf.strings.to_number(tf.strings.split(x, ',')))
    num_vars = next(iter(train_ds)).shape[-1]
    train_xy = tf.stack([x for x in train_ds])
    lb_id = 4
    train_x = tf.gather(train_xy, [i for i in range(num_vars) if i != lb_id], axis=1)
    train_y = train_xy[:, lb_id]
    train_x = tf.expand_dims(train_x, 1)  # for testing

    model = ParVAE(fts=num_vars - 1, emb=K, dim=D, cost=beta)
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])  # loss=mse better than categorical entropy?
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    model.fit(train_x, train_x, epochs=300, batch_size=batch, callbacks=callbacks)
