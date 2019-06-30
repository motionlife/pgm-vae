import os
import argparse
import tensorflow as tf
from model import VqVAE
from baseline import baseline as bl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', required=True, help='target dataset name')
    parser.add_argument('--embedding', '-k', type=int, required=True, help='embedding dictionary size')
    parser.add_argument('--dim', '-d', type=int, required=True, help='embedding dimension')
    parser.add_argument('--batch', '-b', type=int, default=128, help='training batch size')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='number of epochs for training')
    parser.add_argument('--rate', '-r', type=float, default=0.001, help='learning rate')
    parser.add_argument('--cost', '-c', type=float, default=0.25, help='commitment cost')
    parser.add_argument('--ema', '-m', action='store_true', help='using exponential moving average')
    parser.add_argument('--decay', '-g', type=float, default=0.99, help='EMA decay rate')
    parser.add_argument('--seed', '-s', type=int, default=0, help='integer for random seed')
    parser.add_argument('--device', '-u', type=int, default=0, help='which GPU to use, -1 means only use CPU')
    args = parser.parse_args()
    name, K, D, batch_size, epochs, learn_rate, beta, ema, gamma, seed, device = (args.name, args.embedding, args.dim,
                                                                                  args.batch, args.epoch, args.rate,
                                                                                  args.cost, args.ema, args.decay,
                                                                                  args.seed, args.device)
    if device == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # training on cpu
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[device], 'GPU')
        # tf.config.experimental.set_memory_growth(gpus[device], True)  # only grow the memory usage as is needed
    tf.random.set_seed(seed)
    identifier = f"{name}_K-{K}_D-{D}_bs-{batch_size}_epk-{epochs}_lr-{learn_rate}_bta-{beta}_gma-{gamma}_sd-{seed}"
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.curdir, "logs", identifier))]
    num_vars = bl[name]['vars']
    lyr0 = min(num_vars / 1.2, 300)
    lyr1 = min(max(num_vars / 3, 1.2 * D), lyr0)

    def get_data(ds_type):
        ds_xy = tf.data.experimental.CsvDataset(f'trw/{name}.{ds_type}.data', [0.] * num_vars).map(
            lambda *x: tf.stack(x))
        ds_x = tf.stack([x for x in ds_xy.map(lambda x: tf.reshape(tf.tile(x, [num_vars - 1]), [num_vars, -1]))])
        ds_y = tf.stack([y for y in ds_xy.map(lambda x: tf.reverse(x, [0]))])
        return ds_x, ds_y, bl[name][ds_type]

    train_x, train_y, train_size = get_data('train')
    model = VqVAE(units=[lyr0, lyr1], fts=num_vars - 1, dim=D, emb=K, cost=beta, decay=gamma, ema=ema)
    opt = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])  # mse, categorical_crossentropy, binary_crossentropy
    model.fit(train_x, train_x, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    # model.save_weights(log_dir + '/model', save_format='tf')

    # Calculate conditional distribution from training data
    code = model(train_x, code_only=True)
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
    out = f'-train:{tf.reduce_sum(pll_train)} valid:{tf.reduce_sum(pll_valid)} test:{tf.reduce_sum(pll_test)}'
    with open('pll.txt', 'a') as f:
        f.write(identifier + out + '\n')
    print(identifier + out)
