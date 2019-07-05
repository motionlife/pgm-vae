import os
import tensorflow as tf
import numpy as np
from core.model import VqVAE
from baseline import baseline as bl

if __name__ == '__main__':
    # todo: arg parse parameters from command
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # training on cpu
    name = 'nltcs'
    num_vars = bl[name]['vars']
    batch_size = 64
    K = 50
    D = 8
    dense_units = [12, 10]
    epochs = 100
    learn_rate = 0.001
    beta = 0.25
    gamma = 0.99
    seed = 1
    tf.random.set_seed(seed)
    log_dir = os.path.join(os.curdir, '../../DATA_DRIVE/logs',
                           f'{name}_D-{D}_K-{K}_bs-{batch_size}_lr-{learn_rate}_sd-{seed}')
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    ds_size = bl[name]['train']
    train_xy = tf.data.experimental.CsvDataset(f'data/trw/{name}.train.data', [0.] * num_vars).map(
        lambda *x: tf.stack(x)).shuffle(ds_size // 4)
    train_xs = train_xy.map(lambda x: tf.reshape(tf.tile(x, [num_vars - 1]), [num_vars, -1]))
    train_xx = train_xs.map(lambda x: (x, x)).batch(batch_size).prefetch(100)
    train_x = train_xs.batch(batch_size).prefetch(100)
    train_y = train_xy.map(lambda x: tf.reverse(tf.cast(x, tf.int32), [0])).batch(batch_size).prefetch(100)

    model = VqVAE(units=dense_units, fts=num_vars - 1, dim=D, emb=K, cost=beta, decay=gamma)
    opt = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])  # loss=mse better than categorical entropy?
    model.fit(train_xx, epochs=epochs, callbacks=callbacks, shuffle=False)
    model.save_weights(log_dir + '/model', save_format='tf')

    # Calculate distribution from training data
    counter = np.ones((num_vars, K, 2), np.int32)  # Laplace smoothing with a = 1
    for x_batch, y_batch in zip(train_x, train_y):
        encoding_idx = model(x_batch, code_idx_only=True)  # shape=(num_vars, batch_size)
        B, V = y_batch.shape
        for v in range(V):
            for b in range(B):
                counter[v, encoding_idx[v, b], y_batch[b, v]] += 1
    dist = counter / np.sum(counter, axis=-1, keepdims=True)

    # Calculate Pseudo Log-Likelihood
    pll = np.zeros(num_vars)
    for x_batch, y_batch in zip(train_x, train_y):
        encoding_idx = model(x_batch, code_idx_only=True)
        B, V = y_batch.shape
        for v in range(V):
            for b in range(B):
                pll[v] += np.log(dist[v, encoding_idx[v, b], y_batch[b, v]])
    avg_pll = np.sum(pll / ds_size)

    print(f'The total (variable) average PLL is: {avg_pll}')
