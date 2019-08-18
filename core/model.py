# Parameter tying with auto-encoder of discrete latent representation
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Customized tensorflow keras model that can train a pgm parameter tying model and do profiling"""

import tensorflow as tf
from tensorflow.python.keras import Model
from core.dense import FatDense
# from extern.vqvae import VectorQuantizer, VectorQuantizerEMA
from core.quantizer import VectorQuantizerEMA, VectorQuantizer, VectorQuantizerNaive


class VqVAE(Model):
    """A customized model derived from Keras model for batch features training"""

    def __init__(self, units, nvar, dim, k, cost=0.5, decay=0.99, ema=True):
        super(VqVAE, self).__init__(name='vq_vae')
        act = 'selu'
        init = 'he_uniform'
        self.fd0 = FatDense(units[0], activation=act, kernel_initializer=init)
        self.fd1 = FatDense(units[1], activation=act, kernel_initializer=init)
        self.fd2 = FatDense(units[2], activation=act, kernel_initializer=init)
        self.fd3 = FatDense(units[3], activation=act, kernel_initializer=init)
        self.fd4 = FatDense(dim, activation=act, kernel_initializer=init)
        self.vq_layer = VectorQuantizerEMA(embedding_dim=dim, num_embeddings=k, commitment_cost=cost, decay=decay,
                                           num_var=nvar) if ema else VectorQuantizer(embedding_dim=dim,
                                                                                     num_embeddings=k,
                                                                                     commitment_cost=cost,
                                                                                     num_var=nvar)
        # self.vq_layer = VectorQuantizerNaive(dim, commitment_cost=cost, name='naive_vector_quantizer')
        self.fd5 = FatDense(units[3], activation=act, kernel_initializer=init)
        self.fd6 = FatDense(units[2], activation=act, kernel_initializer=init)
        self.fd7 = FatDense(units[1], activation=act, kernel_initializer=init)
        self.fd8 = FatDense(units[0], activation=act, kernel_initializer=init)
        self.fd9 = FatDense(nvar - 1, activation='sigmoid', kernel_initializer='glorot_uniform')
        self.dist = tf.zeros([nvar, k], dtype=tf.float64)

    def call(self, inputs, training=None, code_only=False, fts=None):
        # switch feature and batch dimension
        x = tf.transpose(inputs, [1, 0, 2]) if fts is None else inputs
        x = self.fd0(x, fts=fts)
        x = self.fd1(x, fts=fts)
        x = self.fd2(x, fts=fts)
        x = self.fd3(x, fts=fts)
        x = self.fd4(x, fts=fts)
        x = self.vq_layer(x, training=training, code_only=code_only, fts=fts)
        if not code_only:
            x = self.fd5(x, fts=fts)
            x = self.fd6(x, fts=fts)
            x = self.fd7(x, fts=fts)
            x = self.fd8(x, fts=fts)
            x = self.fd9(x, fts=fts)
            x = tf.transpose(x, [1, 0, 2])
        return x

    # @tf.function
    def count(self, x, y):
        """return the total number of (y=1, x=k) and (y=0, x=k) from data"""

        def fn(e):
            return tf.reduce_sum(tf.boolean_mask(e[0], e[1]), 0)

        b = 200
        n1, n0 = tf.zeros([1]), tf.zeros([1])
        q, r = divmod(y.shape[0], b)
        for i in tf.range(q):
            y_ = tf.transpose(y[i * b:(i + 1) * b, :])
            code = self(x[i * b:(i + 1) * b, :, :], code_only=True)  # shape=(num_vars, batch_size, K)
            n1_ = tf.map_fn(fn, elems=[code, y_], dtype=code.dtype, back_prop=False)
            n0_ = tf.map_fn(fn, elems=[code, 1 - y_], dtype=code.dtype, back_prop=False)
            n1 += n1_
            n0 += n0_
        if r > 0:
            y_ = tf.transpose(y[q * b:q * b + r, :])
            code = self(x[q * b:q * b + r, :, :], code_only=True)  # shape=(num_vars, batch_size, K)
            n1_ = tf.map_fn(fn, elems=[code, y_], dtype=code.dtype, back_prop=False)
            n0_ = tf.map_fn(fn, elems=[code, 1 - y_], dtype=code.dtype, back_prop=False)
            n1 += n1_
            n0 += n0_

        return tf.cast(n1, tf.float64), tf.cast(n0, tf.float64)

    # @tf.function
    def cpt(self, x, y):
        """return the distribution of p(y=1|x=k) from training data"""
        n1, n0 = self.count(x, y)
        return (n1 + 0.8) / (n1 + n0 + 1.6)  # shape=(num_vars, K), Additive(Laplace) smoothing

    # @tf.function
    def pseudo_log_likelihood(self, x, y):
        """calculate the average pseudo log likelihood for input data"""
        lp1 = tf.math.log(self.dist + 1e-10)  # log_p(y=1|x=k)
        lp0 = tf.math.log(1 - self.dist + 1e-10)  # log_p(y=0|x=k)
        n1, n0 = self.count(x, y)
        return tf.reduce_sum(n1 * lp1 + n0 * lp0) / y.shape[0]

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

    def conditional_marginal_log_likelihood(self, x, p1, num_smp, burn_in, verbose=True):
        """calculate the conditional marginal log-likelihood of test data points via gibbs sampling
        Args:
            x: the test data, shape=(batch_size, num_vars)
            p1: number of variables in each partition block, except for the last one
            num_smp: number of total gibbs samples need to be generated
            burn_in: to ignore some number of samples at the beginning
            verbose: whether to print info during each sampling process
        Return:
            the conditional marginal log-likelihood for the batch of data
        """
        batch_size, dim = x.shape
        blocks = tf.cast(tf.math.ceil(dim / p1), tf.int32)
        vol = tf.concat([tf.tile([p1], [blocks - 1]), [dim - p1 * (blocks - 1)]], axis=0)
        bid = tf.range(blocks)
        marker = bid * p1
        state = tf.Variable(tf.tile(tf.expand_dims(x, 0), [blocks, 1, 1]), trainable=False, name='sample_state')
        cnt = tf.Variable(tf.zeros(x.shape), trainable=False, name='y1_counter')

        @tf.function
        def sampling():
            for i in tf.range(num_smp * p1):
                y = marker + tf.math.mod(i, vol)
                xs = tf.map_fn(
                    lambda b: tf.gather(state[b], tf.concat([tf.range(0, y[b]), tf.range(y[b] + 1, dim)], 0), axis=1),
                    bid, state.dtype, back_prop=0)
                prb = self.get_probability(xs, fts=y)
                gibbs = tf.cast(tf.random.uniform([blocks, batch_size], 0, 1) < prb, state.dtype)
                tf.map_fn(lambda b: state[b, :, y[b]].assign(gibbs[b]), bid, state.dtype, back_prop=0)
                if i > burn_in * p1:
                    tf.map_fn(lambda b: cnt[:, y[b]].assign(cnt[:, y[b]] + gibbs[b]), bid, cnt.dtype, back_prop=0)
                if verbose:
                    tf.print(tf.strings.format('# of samples: {}, component: {}', [i // p1, y[0]]))

        sampling()
        valid = num_smp - burn_in
        valid_end = valid * p1 // tf.cast(vol[-1], tf.float32),
        cmll = cnt / tf.concat([tf.ones([1, dim - vol[-1]]) * valid, tf.ones([1, vol[-1]]) * valid_end], 1)
        return tf.reduce_sum(x * tf.math.log(cmll + 1e-10) + (1 - x) * tf.math.log(1 - cmll + 1e-10)) / batch_size


if __name__ == '__main__':
    import timeit

    print('Test function ---> model.conditional_marginal_log_likelihood')
    num_vars = 150
    K = 15
    D = 20
    num_test_data = 5000
    data = tf.cast(tf.random.uniform([num_test_data, num_vars], minval=0, maxval=2, dtype=tf.int32), tf.float32)
    train_x = tf.stack([tf.reshape(tf.tile(x, [num_vars - 1]), [num_vars, -1]) for x in data])
    model = VqVAE(units=[70, 50, 30], nvar=num_vars - 1, dim=D, k=K, cost=0.25, decay=0.99, ema=True)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.fit(train_x, train_x, batch_size=256, epochs=2, verbose=1)
    rnd = tf.random.uniform([num_vars, K], minval=0, maxval=1, dtype=tf.float64)
    model.dist = rnd / tf.reduce_sum(rnd, 1, keepdims=True)

    print(timeit.timeit(
        lambda: print(model.conditional_marginal_log_likelihood(data, p1=num_vars // 12, num_smp=1000, burn_in=100)),
        number=1))
