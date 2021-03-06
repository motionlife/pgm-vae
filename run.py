import os
import argparse
import numpy as np
import random as rdn
import tensorflow as tf
from core.model import VqVAE
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
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose mode when do model fitting and sampling')
    parser.add_argument('--note', '-t', type=str, default='', help='note for other conditions')
    args = parser.parse_args()
    name, K, D, bs, epochs, learn_rate, beta, ema, gamma, seed, device, vb, note = (args.name, args.embedding, args.dim,
    args.batch, args.epoch, args.rate, args.cost, args.ema, args.decay, args.seed, args.device, args.verbose, args.note)
    if device == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # training on cpu
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[device], 'GPU')
        # for g in gpus:
        #     tf.config.experimental.set_memory_growth(g, True)  # only grow the memory usage as is needed
    os.environ['PYTHONHASHSEED'] = '0'
    rdn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    identifier = f"{name}_K-{K}_D-{D}_bs-{bs}_epk-{epochs}_lr-{learn_rate}_bta-{beta}_ema-{ema}_gma-{gamma}_sd-{seed}-{note}"
    log_dir = os.path.join(os.curdir, "logs", "tuning", identifier)
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)]
    n_var = bl[name]['vars']
    # lyr0 = 40  # max(min(n_var / 2, 200), D)
    # lyr1 = 30  # max(min(n_var / 3, lyr0), D)
    # lyr2 = 20  # max(min(n_var / 5, lyr1), D)
    # lyr3 = 15
    idx = tf.constant([i for i in range(n_var ** 2) if i % (n_var + 1) != 0])

    @tf.function
    def make_xs(ys):
        return tf.map_fn(lambda x: tf.reshape(tf.gather(tf.tile(x, [n_var]), idx), [n_var, -1]), ys, back_prop=0)

    def get_data(tvt):
        # todo: design data pipeline for large dataset, below only for dataset less than 4G after transformed
        ds = tf.data.experimental.CsvDataset(f'data/trw/{name}.{tvt}.data', [0.] * n_var).map(lambda *x: tf.stack(x))
        ys = tf.stack([y for y in ds])
        return make_xs(ys), ys

    train_x, train_y = get_data('train')
    model = VqVAE(units=bl[name]['units'], nvar=n_var, dim=D, k=K, cost=beta, decay=gamma, ema=ema)
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # categorical_crossentropy, binary_crossentropy
    model.fit(train_x, train_x, batch_size=bs, epochs=epochs, callbacks=callbacks, verbose=vb)
    # model.save_weights(log_dir + '/model', save_format='tf')

    # get the conditional distribution from training data
    model.dist = model.cpt(train_x, train_y)

    # get pseudo log likelihood for each 3 data set
    test_x, test_y = get_data('test')
    pll_train = model.pseudo_log_likelihood(train_x, train_y)
    pll_valid = model.pseudo_log_likelihood(*get_data('valid'))
    pll_test = model.pseudo_log_likelihood(test_x, test_y)
    # calculate cmll
    # cmll = model.conditional_marginal_log_likelihood(test_y, p1=n_var // 10, num_smp=3000, burn_in=150, verbose=vb)

    # store and print output result
    out = f' pll-train:{pll_train} pll-valid:{pll_valid} pll-test:{pll_test} cmll-test:{1}'
    with open('result.txt', 'a') as f:
        f.write(identifier + out + '\n')
    print(identifier + out)
