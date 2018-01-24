from __future__ import print_function
import tensorflow as tf
from wideresnet import wideresnet
from accumulator import Accumulator
from utils import get_available_gpus, average_gradients
from cifar10 import cifar10_input, NUM_TRAIN, NUM_TEST
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--depth', type=int, default=28)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=20)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--eval_freq', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_gpus', type=int, default=None)
args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
savedir = ('./results/%d_%d' % (args.depth, args.K)) \
        if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)

available_gpus = get_available_gpus()
n_gpus = len(available_gpus) if args.n_gpus is None else args.n_gpus
available_gpus = available_gpus[:n_gpus]
print ('GPUs to be used: ' + str(available_gpus))

batch_size = args.batch_size
n_train_batches = NUM_TRAIN / batch_size
n_test_batches = NUM_TEST / args.batch_size
with tf.device('/cpu:0'):
    x, y = cifar10_input(batch_size, True)
    xs = tf.split(x, n_gpus, axis=0)
    ys = tf.split(y, n_gpus, axis=0)

    x, y = cifar10_input(batch_size, False)
    txs = tf.split(x, n_gpus, axis=0)
    tys = tf.split(y, n_gpus, axis=0)

global_step = tf.train.get_or_create_global_step()
bdrs = [n_train_batches*int(args.n_epochs*r-1) for r in [.3, .6, .8]]
vals = [args.lr*r for r in [1, 0.2, 0.2**2, 0.2**3]]
lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), bdrs, vals)
optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

grads = [0]*n_gpus
nets = [0]*n_gpus
tnets = [0]*n_gpus

for i in range(n_gpus):
    with tf.device(available_gpus[i]), tf.variable_scope(tf.get_variable_scope()) as scope:
        if i > 0:
            scope.reuse_variables()
        nets[i] = wideresnet(xs[i], ys[i], args.depth, args.K,
                10, True, decay=args.decay)
        scope.reuse_variables()
        tnets[i] = wideresnet(txs[i], tys[i], args.depth, args.K,
                10, False, decay=args.decay)
        loss = nets[i]['cent'] + nets[i]['wd']
        grads[i] = optim.compute_gradients(loss)

with tf.device('/cpu:0'):
    cent = tf.add_n([net['cent'] for net in nets])/n_gpus
    acc = tf.add_n([net['acc'] for net in nets])/n_gpus
    tcent = tf.add_n([net['cent'] for net in tnets])/n_gpus
    tacc = tf.add_n([net['acc'] for net in tnets])/n_gpus
    grads = average_gradients(grads)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optim.apply_gradients(grads, global_step=global_step)

def train():
    saver = tf.train.Saver()
    logfile = open(os.path.join(savedir, 'train.log'), 'w', 0)

    train_logger = Accumulator('cent', 'acc')
    train_to_run = [train_op, cent, acc]
    test_logger = Accumulator('cent', 'acc')
    test_to_run = [tcent, tacc]

    argdict = vars(args)
    print(argdict)
    for k, v in argdict.iteritems():
        logfile.write(k + ': ' + str(v) + '\n')
    logfile.write('\n')

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(args.n_epochs):
        line = 'Epoch %d start, learning rate %f' % (i+1, sess.run(lr))
        print(line)
        logfile.write(line + '\n')
        start = time.time()
        train_logger.clear()
        for j in range(n_train_batches):
            train_logger.accum(sess.run(train_to_run))
            if (j+1) % args.print_freq == 0:
                train_logger.print_(header='train', epoch=i+1, it=j+1,
                        time=time.time()-start, logfile=logfile)

        if (i+1) % args.eval_freq == 0:
            test_logger.clear()
            for j in range(n_test_batches):
                test_logger.accum(sess.run(test_to_run))
            test_logger.print_(header='test', epoch=i+1,
                    time=time.time()-start, logfile=logfile)

        print()
        logfile.write('\n')

        if (i+1) % args.save_freq == 0:
            saver.save(sess, os.path.join(savedir, 'model'))

def test():
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(savedir, 'model'))
    logger = Accumulator('cent', 'acc')
    for j in range(n_test_batches):
        logger.accum(sess.run([tcent, tacc]))
    logger.print_(header='test')

if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode %s' % args.mode)
