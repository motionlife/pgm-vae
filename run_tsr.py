import os
import tensorflow as tf
from model import VqVAE
from baseline import baseline

if __name__ == '__main__':
    # todo: arg parse parameters from command
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # training on cpu
    bl = baseline()
    name = 'jester'
    K = 50
    D = 20
    dense_units = [70, 30]
    batch_size = 88
    epochs = 300
    learn_rate = 0.002
    beta = 0.25
    gamma = 0.99
    seed = 2
    tf.random.set_seed(seed)
    log_dir = os.path.join(os.curdir, "logs", f"{name}_D-{D}_K-{K}_bs-{batch_size}_lr-{learn_rate}_sd-{seed}")
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    num_vars = bl[name]['vars']

    def get_data(ds_type):
        ds_xy = tf.data.experimental.CsvDataset(f'trw/{name}.{ds_type}.data', [0.] * num_vars).map(
            lambda *x: tf.stack(x))
        ds_x = tf.stack([x for x in ds_xy.map(lambda x: tf.reshape(tf.tile(x, [num_vars - 1]), [num_vars, -1]))])
        ds_y = tf.stack([y for y in ds_xy.map(lambda x: tf.reverse(x, [0]))])
        return ds_x, ds_y, bl[name][ds_type]

    train_x, train_y, train_size = get_data('train')
    model = VqVAE(units=dense_units, fts=num_vars - 1, dim=D, emb=K, cost=beta, decay=gamma, ema=True)
    opt = tf.keras.optimizers.Adam(lr=learn_rate)
    # loss=mse, categorical_crossentropy, binary_crossentropy?
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['mae'])
    model.fit(train_x, train_x, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    model.save_weights(log_dir + '/model', save_format='tf')

    # Calculate conditional distribution from training data
    code = model(train_x, code_only=True)  # shape=(ds_size, num_vars, K)
    cnt1 = tf.stack([tf.reduce_sum(tf.boolean_mask(code[:, i, :], train_y[:, i]), 0) for i in range(num_vars)])
    cnt0 = tf.stack([tf.reduce_sum(tf.boolean_mask(code[:, i, :], 1 - train_y[:, i]), 0) for i in range(num_vars)])
    dist = tf.cast(cnt1 + 1, tf.float64) / tf.cast(cnt0 + cnt1 + 2, tf.float64)  # Additive (Laplace) smoothing
    logP1 = tf.math.log(dist)
    logP0 = tf.math.log(1 - dist)

    # Calculate Pseudo Log-Likelihood
    def get_pll(ds_x, ds_y, ds_size):
        enc = model(ds_x, code_only=True)
        n1 = tf.stack([tf.reduce_sum(tf.boolean_mask(enc[:, i, :], ds_y[:, i]), 0) for i in range(num_vars)])
        n0 = tf.stack([tf.reduce_sum(tf.boolean_mask(enc[:, i, :], 1 - ds_y[:, i]), 0) for i in range(num_vars)])
        return (tf.cast(n1, tf.float64) * logP1 + tf.cast(n0, tf.float64) * logP0) / ds_size

    pll_train = get_pll(train_x, train_y, train_size)
    pll_valid = get_pll(*get_data('valid'))
    pll_test = get_pll(*get_data('test'))
    print(f'The total (train) average PLL is: {tf.reduce_sum(pll_train)}')
    print(f'The total (valid) average PLL is: {tf.reduce_sum(pll_valid)}')
    print(f'The total (test) average PLL is: {tf.reduce_sum(pll_test)}')
