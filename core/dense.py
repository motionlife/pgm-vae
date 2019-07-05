# Parameter tying with auto-encoder of discrete latent representation
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""Customized keras dense layer perform multiple independent nn calculation"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import activations as acts
from tensorflow.python.keras import constraints as ctr
from tensorflow.python.keras import initializers as init
from tensorflow.python.keras import regularizers as reg


class FatDense(Layer):
    """ A N-D layer consist of many independent keras Dense layers

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Input shape:
      N-D tensor with shape: `(..., batch_size, input_dim)`.
      The most common situation would be
      a 3D input with shape `(net_num, batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(..., batch_size, units)`.
      For instance, for a 3D input with shape `(net_num, batch_size, input_dim)`,
      the output would have shape `(net_num, batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(FatDense, self).__init__(activity_regularizer=reg.get(activity_regularizer), **kwargs)
        self.units = int(units)
        self.activation = acts.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = init.get(kernel_initializer)
        self.bias_initializer = init.get(bias_initializer)
        self.kernel_regularizer = reg.get(kernel_regularizer)
        self.bias_regularizer = reg.get(bias_regularizer)
        self.kernel_constraint = ctr.get(kernel_constraint)
        self.bias_constraint = ctr.get(bias_constraint)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3:
            raise ValueError("The input tensor must be rank of 3")
        last_dim = input_shape[-1]
        num_fts = input_shape[0]
        self.kernel = self.add_weight(
            name='kernel',
            shape=[num_fts, last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[num_fts, 1, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, fts=None):
        kernel = self.kernel if fts is None else tf.gather(self.kernel, fts, axis=0)
        outputs = tf.matmul(inputs, kernel)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': acts.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': init.serialize(self.kernel_initializer),
            'bias_initializer': init.serialize(self.bias_initializer),
            'kernel_regularizer': reg.serialize(self.kernel_regularizer),
            'bias_regularizer': reg.serialize(self.bias_regularizer),
            'activity_regularizer': reg.serialize(self.activity_regularizer),
            'kernel_constraint': ctr.serialize(self.kernel_constraint),
            'bias_constraint': ctr.serialize(self.bias_constraint)
        }
        base_config = super(FatDense, self).get_config()
        return base_config.update(config)
