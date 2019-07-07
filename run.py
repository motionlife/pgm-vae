import os
import argparse
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
        ds_xy = tf.data.experimental.CsvDataset(f'data/trw/{name}.{ds_type}.data', [0.] * num_vars).map(
            lambda *x: tf.stack(x))
        # todo: don't change the order of x's feature!!!
        ds_x = tf.stack([x for x in ds_xy.map(lambda x: tf.reshape(tf.tile(x, [num_vars - 1]), [num_vars, -1]))])
        ds_y = tf.stack([y for y in ds_xy.map(lambda x: tf.reverse(x, [0]))])
        return ds_x, ds_y


    train_x, train_y = get_data('train')
    model = VqVAE(units=[lyr0, lyr1], fts=num_vars - 1, dim=D, emb=K, cost=beta, decay=gamma, ema=ema)
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])  # categorical_crossentropy, binary_crossentropy
    model.fit(train_x, train_x, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=1)
    # model.save_weights(log_dir + '/model', save_format='tf')

    # get the conditional distribution from training data
    model.dist = model.cpt(train_x, train_y)

    # get pseudo log likelihood for each 3 data set
    valid_x, valid_y = get_data('valid')
    test_x, test_y = get_data('test')
    pll_train = model.pseudo_log_likelihood(train_x, train_y)
    pll_valid = model.pseudo_log_likelihood(valid_x, valid_y)
    pll_test = model.pseudo_log_likelihood(test_x, test_y)

    # store and print output result
    out = f'pll-train:{pll_train} pll-valid:{pll_valid} pll-test:{pll_test}'
    with open('pll.txt', 'a') as f:
        f.write(identifier + out + '\n')
    print(identifier + out)

    cmll = model.conditional_marginal_log_likelihood(tf.reshape(test_x, [-1, num_vars * (num_vars - 1)])[:, :num_vars],
                                                     q_size=100, num_smp=2000, burn_in=80)
