# Parameter tying with auto-encoder of discrete latent representation
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Customized tensorflow keras model that can train a pgm parameter tying model and do profiling"""

import tensorflow as tf
from tensorflow.python.keras import Model
from core.dense import FatDense
from core.vqvae import VectorQuantizerEMA, VectorQuantizer


class VqVAE(Model):
    """A customized model derived from Keras model for batch features training"""

    def __init__(self, units, fts, dim, emb, cost=0.25, decay=0.99, ema=True):
        super(VqVAE, self).__init__(name='vq_vae')
        # regularization: dropout layer or L2 regularizer
        self.fd1 = FatDense(units[0], activation='relu')
        self.fd2 = FatDense(units[1], activation='relu')
        self.fd3 = FatDense(dim, activation='relu')
        self.vq_layer = VectorQuantizerEMA(embedding_dim=dim, num_embeddings=emb, commitment_cost=cost,
                                           decay=decay) if ema else VectorQuantizer(embedding_dim=dim,
                                                                                    num_embeddings=emb,
                                                                                    commitment_cost=cost)
        self.fd4 = FatDense(units[1], activation='relu')
        self.fd5 = FatDense(units[0], activation='relu')
        self.fd6 = FatDense(fts, activation='sigmoid')  # any better activation with [0,1] output?
        self.dist = tf.zeros([fts + 1, emb], dtype=tf.float64)

    def call(self, inputs, code_only=False, fts=None):
        # switch feature and batch dimension
        x = tf.transpose(inputs, [1, 0, 2]) if fts is None else inputs
        x = self.fd1(x, fts=fts)
        x = self.fd2(x, fts=fts)
        x = self.fd3(x, fts=fts)
        x = self.vq_layer(x, code_only=code_only, fts=fts)
        if not code_only:
            x = self.fd4(x, fts=fts)
            x = self.fd5(x, fts=fts)
            x = self.fd6(x, fts=fts)
            x = tf.transpose(x, [1, 0, 2])
        return x

    @tf.function
    def count(self, x, y):
        """return the total number of (y=1, x=k) and (y=0, x=k) from data"""
        y = tf.transpose(y)

        def fn(e): return tf.reduce_sum(tf.boolean_mask(e[0], e[1]), 0)

        code = self(x, code_only=True)  # shape=(num_vars, batch_size, K)
        n1 = tf.map_fn(fn, elems=[code, y], dtype=code.dtype, back_prop=False)
        n0 = tf.map_fn(fn, elems=[code, 1 - y], dtype=code.dtype, back_prop=False)
        return n1, n0

    @tf.function
    def cpt(self, x, y):
        """return the distribution of p(y=1|x=k) from training data"""
        n1, n0 = self.count(x, y)
        n1 = tf.cast(n1 + 1, tf.float64)  # shape=(num_vars, K), Additive(Laplace) smoothing
        n0 = tf.cast(n0 + 1, tf.float64)
        return n1 / (n1 + n0)

    @tf.function
    def pseudo_log_likelihood(self, x, y):
        """calculate the average pseudo log likelihood for input data"""
        lp1 = tf.math.log(self.dist)  # log_p(y=1|x=k)
        lp0 = tf.math.log(1 - self.dist)  # log_p(y=0|x=k)
        n1, n0 = self.count(x, y)
        return tf.reduce_sum(tf.cast(n1, tf.float64) * lp1 + tf.cast(n0, tf.float64) * lp0) / y.shape[0]

    @tf.function
    def get_probability(self, x, fts=None):
        """get the conditional probability (y_i=1) of inputs x from this model's conditional distribution
        Args:
            x: the test data, must be binary, shape=(num_selected_fts, batch_size, num_vars-1)
            fts: the indices of selected features (corresponding to their own neural net)
        Return: a tensor contains conditional probability, shape=(num_selected_fts, batch_size)
        """
        enc_idx = self(x, code_only=True, fts=fts)  # shape=(num_selected_fts, batch_size)
        prb = tf.cast(tf.gather(self.dist, fts, axis=0), tf.float32)
        return tf.gather(prb, enc_idx, axis=1, batch_dims=1)

    def conditional_marginal_log_likelihood(self, x, q_size, num_smp, burn_in, verbose=False):
        """calculate the conditional marginal log-likelihood of test data points via gibbs sampling
        Args:
            x: the test data, shape=(batch_size, num_vars)
            q_size: number of variables in each partition block, except for the last one
            num_smp: number of total gibbs samples need to be generated
            burn_in: to ignore some number of samples at the beginning
            verbose: whether to print info during each sampling process
        Return:
            the conditional marginal log-likelihood for the batch of data
        """
        batch_size, dim = x.shape
        blocks = tf.cast(tf.math.ceil(dim / q_size), tf.int32)
        vol = tf.concat([tf.tile([q_size], [blocks - 1]), [dim - q_size * (blocks - 1)]], axis=0)
        bid = tf.range(blocks)
        mark = bid * q_size
        state = tf.Variable(tf.tile(tf.expand_dims(x, 0), [blocks, 1, 1]), trainable=False, name='samples')
        cnt = tf.Variable(tf.zeros(x.shape), trainable=False, name='y1_counter')
        @tf.function
        def sampling():
            for i in tf.range(num_smp * q_size):
                y = mark + tf.math.mod(i, vol)
                xs = tf.map_fn(
                    lambda b: tf.gather(state[b], tf.concat([tf.range(0, y[b]), tf.range(y[b] + 1, dim)], 0), axis=1),
                    bid, state.dtype, back_prop=0)
                prb = self.get_probability(xs, fts=y)
                gibbs = tf.cast(tf.random.uniform([blocks, batch_size], 0, 1) < prb, state.dtype)
                tf.map_fn(lambda b: state[b, :, y[b]].assign(gibbs[b]), bid, state.dtype, back_prop=0)
                if i > burn_in * q_size:
                    tf.map_fn(lambda b: cnt[:, y[b]].assign(cnt[:, y[b]] + gibbs[b]), bid, cnt.dtype, back_prop=0)
                if verbose:
                    tf.print(tf.strings.format('# of samples: {}, component: {}', [tf.math.ceil(i / q_size), y[0]]))

        sampling()
        cml = cnt / tf.concat([tf.ones([1, dim - vol[-1]]) * (num_smp - burn_in),
                               tf.ones([1, vol[-1]]) * (num_smp - burn_in) * tf.cast(q_size / vol[-1], tf.float32)], 1)
        return tf.reduce_sum(x * tf.math.log(cml + 1e-10) + (1 - x) * tf.math.log(1 - cml + 1e-10)) / batch_size


if __name__ == '__main__':
    import timeit
    print('Test function ---> model.conditional_marginal_log_likelihood')
    num_vars = 150
    data = tf.cast(tf.random.uniform([5000, num_vars], minval=0, maxval=2, dtype=tf.int32), tf.float32)
    train_x = tf.stack([tf.reshape(tf.tile(x, [num_vars - 1]), [num_vars, -1]) for x in data])
    model = VqVAE(units=[70, 30], fts=num_vars - 1, dim=20, emb=40, cost=0.25, decay=0.99, ema=True)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.fit(train_x, train_x, batch_size=128, epochs=2, verbose=1)
    model.dist = tf.random.uniform([num_vars, 40], minval=0, maxval=1, dtype=tf.float64)

    print(timeit.timeit(
      lambda: print(model.conditional_marginal_log_likelihood(data, q_size=10, num_smp=100, burn_in=10, verbose=True)),
      number=1))
