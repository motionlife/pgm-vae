import os
import tensorflow as tf
import numpy as np
from model import VqVAE
from baseline import baseline

if __name__ == '__main__':
    # todo: arg parse parameters from command
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # training on cpu
    bl = baseline()
    name = 'netflix'
    K = 30
    D = 15
    dense_units = [50, 30]
    batch_size = 64
    epochs = 200
    learn_rate = 0.01
    beta = 0.25
    gamma = 0.99
    seed = 1
    tf.random.set_seed(seed)
    log_dir = os.path.join(os.curdir, "logs", f"{name}_D-{D}_K-{K}_bs-{batch_size}_lr-{learn_rate}_sd-{seed}")
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]
    num_vars = bl[name]['vars']

    def get_data(ds_type):
        ds_xy = tf.data.experimental.CsvDataset(f'trw/{name}.{ds_type}.data', [0.] * num_vars).map(
            lambda *x: tf.stack(x))
        ds_x = tf.stack([x for x in ds_xy.map(lambda x: tf.reshape(tf.tile(x, [num_vars - 1]), [num_vars, -1]))])
        ds_y = tf.stack([y for y in ds_xy.map(lambda x: tf.reverse(tf.cast(x, tf.int32), [0]))])
        return ds_x, ds_y, bl[name][ds_type]

    train_x, train_y, train_size = get_data('train')
    model = VqVAE(units=dense_units, fts=num_vars - 1, dim=D, emb=K, cost=beta, decay=gamma, ema=True)
    opt = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])  # loss=mse better than categorical entropy?
    model.fit(train_x, train_x, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    model.save_weights(log_dir + '/model', save_format='tf')

    # Calculate distribution from training data
    counter = np.ones((num_vars, K, 2), np.int32)  # Laplace smoothing with a = 1
    encoding_idx = model(train_x, code_idx_only=True)  # shape=(num_vars, batch_size)
    for v in range(num_vars):
        for i in range(train_size):
            counter[v, encoding_idx[v, i], train_y[i, v]] += 1
    dist = counter / np.sum(counter, axis=-1, keepdims=True)

    # Calculate Pseudo Log-Likelihood
    def get_pll(ds_x, ds_y, ds_size):
        pll = np.zeros(num_vars)
        indices = model(ds_x, code_idx_only=True)
        for n in range(num_vars):
            for j in range(ds_size):
                pll[n] += np.log(dist[n, indices[n, j], ds_y[j, n]])
        return pll / ds_size

    pll_train = get_pll(train_x, train_y, train_size)  # np.sum(pll / train_size)
    pll_valid = get_pll(*get_data('valid'))
    pll_test = get_pll(*get_data('test'))
    print(f'The total (train) average PLL is: {np.sum(pll_train)}')
    print(f'The total (valid) average PLL is: {np.sum(pll_valid)}')
    print(f'The total (test) average PLL is: {np.sum(pll_test)}')
