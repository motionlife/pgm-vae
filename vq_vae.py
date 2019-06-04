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


# noinspection PyAttributeOutsideInit
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
            shape = tf.TensorShape([self._embedding_dim, self._num_embeddings])
        initializer = tf.keras.initializers.lecun_normal(seed=7)  # he_normal(seed=7)?
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

        flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])
        distances = (tf.reduce_sum(flat_inputs ** 2, 1, keepdims=True)
                     - 2 * tf.matmul(flat_inputs, self._embeddings)
                     + tf.reduce_sum(self._embeddings ** 2, 0, keepdims=True))

        encoding_indices = tf.argmin(distances, 1)
        # encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = tf.nn.embedding_lookup(tf.transpose(self._embeddings.read_value()), encoding_indices)
        # calculate quantization layer loss
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)  # commitment loss
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        self.add_loss(loss)
        # make sure grad(Zq(x)) = grad(Ze(x))
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        # avg_probs = tf.reduce_mean(encodings, 0)
        # perplexity = tf.math.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10))) # perplexity = exp(H(p))
        # self._pkgs = {'quantized': quantized,
        #               'loss': loss,
        #               'perplexity': perplexity,
        #               'encodings': encodings,
        #               'encoding_indices': encoding_indices, }

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
    """A customized model derived from Keras model"""

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='vq_vae_model')
        self.num_classes = num_classes
        # Define your layers here.
        self.dense_1 = Dense(32, activation='relu')
        self.dense_2 = Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        # dens1 = Dense(30, activation='relu')(xin)
        # dens2 = Dense(D, activation='relu')(dens1)
        # vq_layer = VQLayer(embedding_dim=D, num_embeddings=K, commitment_cost=beta)(dens2)
        # dens3 = Dense(30, activation='relu')(vq_layer)
        # outputs = Dense(70, activation='relu')(dens3)
        # model = Model(inputs=xin, outputs=outputs, name='vae')
        # model.add_loss(vq_layer.pkgs['loss'])
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


if __name__ == '__main__':
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # using cpu
    log_dir = os.path.join(os.path.join(os.curdir, "logs"), time.strftime("run_%Y_%m_%d-%H_%M_%S"))

    # test layer
    batch = 100
    D = 8
    K = 70
    beta = 0.2

    train_ds = tf.data.TextLineDataset('trw/nltcs.ts.data') \
        .map(lambda x: tf.strings.to_number(tf.strings.split(x, ',')))
    num_vars = next(iter(train_ds)).shape[-1]
    train_xy = tf.stack([x for x in train_ds])
    lb_id = 0
    train_x = tf.gather(train_xy, [i for i in range(num_vars) if i != lb_id], axis=1)
    train_y = train_xy[:, lb_id]
    # todo: use dropout layer to do regularization
    model = tf.keras.Sequential([
        Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
        Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
        Dense(D, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
        VQLayer(embedding_dim=D, num_embeddings=K, commitment_cost=beta),
        Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
        Dense(12, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
        Dense(num_vars - 1, activation='sigmoid'),
    ])  # make sure the output of the model is [0,1]
    opt = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=['mse'])  # loss=mse better than categorical entropy?
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    model.fit(train_x, train_x, epochs=500, batch_size=batch, callbacks=callbacks)
