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

    @tf.function
    def call(self, inputs, code_only=False, fts=None):
        x = tf.transpose(inputs, [1, 0, 2])  # switch feature and batch dimension
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
    def cpt(self, x, y):
        batch_size, num_vars = y.shape
        py1 = tf.cast(tf.expand_dims(tf.reduce_sum(y, 0) + 1, 1), tf.float64) / tf.cast(batch_size + 2, tf.float64)
        y = tf.transpose(y)
        code = self(x, code_only=True)  # shape=(num_vars, batch_size, K)
        n1 = tf.map_fn(lambda e: tf.reduce_sum(tf.boolean_mask(e[0], e[1]), 0), elems=[code, y], dtype=code.dtype)
        n0 = tf.map_fn(lambda e: tf.reduce_sum(tf.boolean_mask(e[0], 1 - e[1]), 0), elems=[code, y], dtype=code.dtype)
        n1 = tf.cast(n1 + 1, tf.float64)   # Additive(Laplace) smoothing
        n0 = tf.cast(n0 + 1, tf.float64)
        # p(y=1|x=k) = p(x=k,y=1)/p(x=k) = p(x=k|y=1)*p(y=1)/p(x=k)
        self.dist = n1 * py1 / (n1 * py1 + n0 * (1 - py1))  # shape=(num_vars, K)

    @tf.function
    def pseudo_log_likelihood(self, x, y):
        batch_size, num_vars = y.shape
        lp1 = tf.math.log(self.dist)
        lp0 = tf.math.log(1 - self.dist)
        y = tf.transpose(y)
        code = self(x, code_only=True)
        n1 = tf.map_fn(lambda e: tf.reduce_sum(tf.boolean_mask(e[0], e[1]), 0), elems=[code, y], dtype=code.dtype)
        n0 = tf.map_fn(lambda e: tf.reduce_sum(tf.boolean_mask(e[0], 1 - e[1]), 0), elems=[code, y], dtype=code.dtype)
        return tf.reduce_sum(tf.cast(n1, tf.float64) * lp1 + tf.cast(n0, tf.float64) * lp0) / batch_size

    @tf.function
    def get_probability(self, x, fts=None):
        """get the conditional probability (y_i=1) of inputs x from this model's conditional distribution
        Args:
            x: the test data, shape=(batch_size, num_selected_vars, num_vars-1)
            fts: the indices of selected features (corresponding to their own neural net)
        Return: a tensor contains conditional probability, shape=(batch_size, num_selected_fts)
        """
        enc_idx = self(x, code_only=True, fts=fts)  # shape=(num_selected_vars, batch_size)
        prb = tf.cast(tf.gather(self.dist, fts, axis=0), tf.float32)
        prb = tf.gather(prb, enc_idx, axis=1, batch_dims=1)
        return tf.transpose(prb)  # shape=(batch_size, num_selected_vars)

    @tf.function
    def conditional_marginal_log_likelihood(self, x, q_size, gibbs_samples, burn_in):
        """calculate the conditional marginal log-likelihood of test data points via gibbs sampling
        Args:
            x: the test data, shape=(batch_size, num_vars)
            q_size: number of variables in each partition block, except for the last one
            gibbs_samples: number of iteration for gibbs sampling
            burn_in: to ignore some number of samples at the beginning
        Return:
            the conditional marginal log-likelihood for the batch of data
        """
        bs, dim = x.shape
        blk = tf.cast(tf.math.ceil(dim / q_size), tf.int32)
        blk_vol = tf.concat([tf.tile([q_size], [blk - 1]), [dim - q_size * (blk - 1)]], axis=0)
        mark = tf.range(blk) * q_size
        idx = tf.stack([[i for i in tf.range(dim) if tf.not_equal(i, j)] for j in tf.range(dim)])
        idx = tf.tile(tf.expand_dims(idx, 0), [bs, 1, 1])  # too large!!!
        smp = tf.Variable(tf.tile(tf.expand_dims(x, 1), [1, blk, 1]), trainable=False, name='samples')
        count = tf.Variable(tf.zeros([bs, dim]), trainable=False, name='y1_count')
        for i in tf.range(gibbs_samples * q_size):
            yid = mark + tf.math.mod(i, blk_vol)
            xs = tf.gather(smp, tf.gather(idx, yid, axis=1), axis=2, batch_dims=2)  # too expensive!!!
            prb = self.get_probability(xs, fts=dim - 1 - yid)  # todo: correct this after data munging
            gibbs = tf.cast(tf.random.uniform([bs, blk], 0, 1) < prb, smp.dtype)
            for b in tf.range(blk):
                smp[:, b, yid[b]].assign(gibbs[:, b])
            if i > burn_in * q_size:
                for b in tf.range(blk):
                    count[:, yid[b]].assign_add(gibbs[:, b])
            tf.print('generated ith component')

        cmll = 0.
        for b in tf.range(blk):
            cmll += tf.reduce_sum(tf.math.log((count[:, b, mark[b]:mark[b] + blk_vol[b]]) / (gibbs_samples - burn_in)))

        return cmll / bs


if __name__ == '__main__':
    print('test conditional_marginal_log_likelihood')
