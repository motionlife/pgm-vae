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


class VQLayer(Layer):
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
        super(VQLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        The __call__ method of layer will automatically run build the first time it is called.
        Create trainable layer weights (embedding dictionary) here
        """
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], self._embedding_dim), [input_shape[-1]])]):
            shape = tf.TensorShape([input_shape[0], self._embedding_dim, self._num_embeddings])
        initializer = tf.keras.initializers.he_uniform(seed=7)  # GlorotUniform(seed=7)?
        self._embeddings = self.add_weight(name='embeddings',
                                           shape=shape,
                                           initializer=initializer,
                                           trainable=True)
        # Make sure to call the `build` method at the end
        super(VQLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        """ Define the forward computation pass """
        if training:
            print("Training process begin...")

        distances = (tf.reduce_sum(inputs ** 2, 2, keepdims=True)
                     - 2 * tf.matmul(inputs, self._embeddings)
                     + tf.reduce_sum(self._embeddings ** 2, 1, keepdims=True))

        enc_idx = tf.argmin(distances, 2)
        # encodings = tf.one_hot(enc_idx, self._num_embeddings)
        quantized = tf.gather(tf.transpose(self._embeddings.read_value(), [0, 2, 1]), enc_idx, axis=1, batch_dims=1)
        #  tf.compat.v1.batch_gather(tf.transpose(emb,[0,2,1]), idx)
        # calculate quantization layer loss
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)  # commitment loss
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        self.add_loss(loss)

        quantized = inputs + tf.stop_gradient(quantized - inputs)  # grad(Zq(x)) = grad(Ze(x))
        # avg_probs = tf.reduce_mean(encodings, 0)
        # perplexity = tf.math.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10))) # perplexity = exp(H(p))
        # self._pkgs = {'quantized': quantized,
        #               'loss': loss,
        #               'perplexity': perplexity,
        #               'encodings': encodings,
        #               'enc_idx': enc_idx, }
        return quantized

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def pkgs(self):
        return self._pkgs

    @staticmethod
    def compute_output_shape(input_shape):
        # return tf.TensorShape(input_shape)
        return input_shape

    # Optionally, a layer can be serialized by implementing the get_config method and the from_config class method.
    def get_config(self):
        base_config = super(VQLayer, self).get_config()
        # base_config['num_embeddings'] = self._num_embeddings
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MyModel(Model):
    """A customized model derived from Keras model for jumbo training"""

    def __init__(self, fts=15, emb=30, dim=8, cost=0.25):
        super(MyModel, self).__init__(name='vq_vae_model')
        # todo: try dropout layer to do regularization
        self.dense_1 = ParallelDense(12, activation='relu')
        self.dense_2 = ParallelDense(10, activation='relu')
        self.dense_3 = ParallelDense(dim, activation='relu')
        self.vq_layer = VQLayer(embedding_dim=dim, num_embeddings=emb, commitment_cost=cost)
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
    lb_id = 1
    train_x = tf.gather(train_xy, [i for i in range(num_vars) if i != lb_id], axis=1)
    train_y = train_xy[:, lb_id]
    train_x = tf.expand_dims(train_x, 1) # for testing

    model = MyModel(fts=num_vars - 1, emb=K, dim=D, cost=beta)
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])  # loss=mse better than categorical entropy?
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    model.fit(train_x, train_x, epochs=500, batch_size=batch, callbacks=callbacks)
