# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937."""
from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import moving_averages

import tensorflow as tf


class VectorQuantizer(base.Module):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, num_var, dtype=tf.float32,
                 name='vector_quantizer'):
        super(VectorQuantizer, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        embedding_shape = [num_var, embedding_dim, num_embeddings]
        initializer = initializers.VarianceScaling(distribution='uniform')
        self.embeddings = tf.Variable(initializer(embedding_shape, dtype), name='embeddings')

    def __call__(self, inputs, training=None, code_only=False, fts=None):
        w = self.embeddings if fts is None else tf.gather(self.embeddings, fts, axis=0)
        distances = (tf.reduce_sum(inputs ** 2, 2, keepdims=True)
                     - 2 * tf.matmul(inputs, w)
                     + tf.reduce_sum(w ** 2, 1, keepdims=True))

        encoding_indices = tf.argmax(- distances, 2)
        if code_only:
            loss = 0.
            output = tf.one_hot(encoding_indices, self.num_embeddings) if fts is None else encoding_indices
        else:
            quantized = tf.gather(tf.transpose(w, [0, 2, 1]), encoding_indices, axis=1, batch_dims=1)
            e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
            q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            output = inputs + tf.stop_gradient(quantized - inputs)

        return output, loss


class VectorQuantizerEMA(base.Module):

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay, num_var,
                 epsilon=1e-5, dtype=tf.float32, name='vector_quantizer_ema'):
        super(VectorQuantizerEMA, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        if not 0 <= decay <= 1:
            raise ValueError('decay must be in range [0, 1]')
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        embedding_shape = [num_var, embedding_dim, num_embeddings]
        initializer = initializers.VarianceScaling(distribution='uniform')
        self.embeddings = tf.Variable(initializer(embedding_shape, dtype), name='embeddings')

        self.ema_cluster_size = moving_averages.ExponentialMovingAverage(decay=self.decay, name='ema_cluster_size')
        self.ema_cluster_size.initialize(tf.zeros([num_var, num_embeddings], dtype=dtype))

        self.ema_dw = moving_averages.ExponentialMovingAverage(decay=self.decay, name='ema_dw')
        self.ema_dw.initialize(self.embeddings)

    def __call__(self, inputs, training=None, code_only=False, fts=None):
        w = self.embeddings if fts is None else tf.gather(self.embeddings, fts, axis=0)
        distances = (tf.reduce_sum(inputs ** 2, 2, keepdims=True)
                     - 2 * tf.matmul(inputs, w)
                     + tf.reduce_sum(w ** 2, 1, keepdims=True))

        encoding_indices = tf.argmax(- distances, 2)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        if code_only:
            loss = 0.
            output = encodings if fts is None else encoding_indices
        else:
            quantized = tf.gather(tf.transpose(w, [0, 2, 1]), encoding_indices, axis=1, batch_dims=1)
            e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
            if training:
                updated_ema_cluster_size = self.ema_cluster_size(tf.reduce_sum(encodings, axis=1))
                dw = tf.matmul(inputs, encodings, transpose_a=True)
                updated_ema_dw = self.ema_dw(dw)
                n = tf.reduce_sum(updated_ema_cluster_size, axis=1, keepdims=True)
                updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
                normalised_updated_ema_w = (updated_ema_dw / tf.expand_dims(updated_ema_cluster_size, 1))
                w.assign(normalised_updated_ema_w)
                loss = self.commitment_cost * e_latent_loss
            else:
                loss = self.commitment_cost * e_latent_loss
            output = inputs + tf.stop_gradient(quantized - inputs)

        return output, loss
