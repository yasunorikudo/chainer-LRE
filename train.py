#!/usr/bin/env python

from __future__ import print_function

import argparse
import copy
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(784, n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units, n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units, n_out)  # n_units -> n_out

    def __call__(self, x):
        return self.forward(x)[-1]

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return [h1, h2, h3]


class LRE_Updator(training.StandardUpdater):

    def update_core(self):
        batch = self._iterators['main'].next()
        x, t = self.converter(batch, self.device)
        batchsize = len(batch)

        batch_val = self._iterators['val'].next()
        x_val, t_val = self.converter(batch_val, self.device)
        batchsize_val = len(batch_val)

        opt_model = self._optimizers['main']
        model = opt_model.target

        opt_model_tmp = copy.deepcopy(opt_model)
        model_tmp = opt_model_tmp.target

        weight = L.Linear(batchsize, 1, nobias=True, initialW=1/batchsize)
        weight.to_gpu()

        ys = model_tmp.predictor.forward(x)
        loss_f = F.sum(weight(F.softmax_cross_entropy(ys[-1], t, reduce='no')[None]))
        model_tmp.cleargrads()
        loss_f.backward(retain_grad=True)
        opt_model_tmp.update()

        ys_val = model_tmp.predictor.forward(x_val)
        loss_g = F.softmax_cross_entropy(ys_val[-1], t_val)
        model_tmp.cleargrads()
        loss_g.backward(retain_grad=True)

        # Eq 12 in https://arxiv.org/abs/1803.09050
        # TODO(yasunorikudo): Faster implementation.
        w = model.xp.zeros(batchsize, dtype=np.float32)
        for i in range(batchsize):
            w[i] += ((x_val * x[i]).sum(axis=1) * (ys_val[0].grad * ys[0].grad[i]).sum(axis=1)).sum()
            for l in range(1, len(ys)):
                w[i] += ((ys_val[l-1].data * ys[l-1].data[i]).sum(axis=1) * (ys_val[l].grad * ys[l].grad[i]).sum(axis=1)).sum()
        w[w < 0] = 0
        if w.sum() != 0:
            w /= w.sum()
        weight.W.data[:] = w[None]

        y = model.predictor(x)
        loss_f2 = F.sum(weight(F.softmax_cross_entropy(y, t, reduce='no')[None]))
        model.cleargrads()
        loss_f2.backward()
        opt_model.update()
        chainer.report({'loss': loss_f2, 'accuracy': F.accuracy(y, t)}, model)


class ClassWeightUpdator(training.StandardUpdater):

    def update_core(self):
        batch = self._iterators['main'].next()
        x, t = self.converter(batch, self.device)
        batchsize = len(batch)

        optimizer = self._optimizers['main']
        model = optimizer.target

        w = np.empty((1, batchsize), dtype='f')
        w[0, cuda.to_cpu(t) == 4] = 1 - 50 / 5000
        w[0, cuda.to_cpu(t) == 9] = 1 - 4950 / 5000
        weight = L.Linear(batchsize, 1, nobias=True, initialW=w)
        weight.to_gpu()

        y = model.predictor(x)
        loss = F.sum(weight(F.softmax_cross_entropy(y, t, reduce='no')[None]))
        model.cleargrads()
        loss.backward()
        optimizer.update()

        chainer.report({'loss': loss, 'accuracy': F.accuracy(y, t)}, model)


def get_imbalanced_mnist(n_train_images={4: 50, 9: 4950},
                         n_val_images_per_class=5, seed=1701):
    assert isinstance(n_train_images, dict)
    assert isinstance(n_val_images_per_class, int)

    train, test = chainer.datasets.get_mnist()
    x, t = chainer.dataset.concat_examples(train)
    train_images, train_labels = None, None
    val_images, val_labels = None, None
    for cls, num in n_train_images.items():
        indices = np.where(t == cls)[0]
        assert len(indices) >= num
        perm = np.random.RandomState(seed=seed).permutation(len(indices))
        train_indices = indices[perm][:num]
        val_indices = indices[perm][:n_val_images_per_class]
        train_images = x[train_indices] if train_images is None else np.concatenate((train_images, x[train_indices]), axis=0)
        train_labels = t[train_indices] if train_labels is None else np.concatenate((train_labels, t[train_indices]), axis=0)
        val_images = x[val_indices] if val_images is None else np.concatenate((val_images, x[val_indices]), axis=0)
        val_labels = t[val_indices] if val_labels is None else np.concatenate((val_labels, t[val_indices]), axis=0)

    x, t = chainer.dataset.concat_examples(test)
    test_indices = None
    for cls in n_train_images.keys():
        ind = np.where(t == cls)[0]
        test_indices = ind if test_indices is None else np.concatenate((test_indices, ind), axis=0)
    test_images = x[test_indices]
    test_labels = t[test_indices]

    train = chainer.datasets.tuple_dataset.TupleDataset(train_images, train_labels)
    val = chainer.datasets.tuple_dataset.TupleDataset(val_images, val_labels)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_images, test_labels)

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--update_algo', '-a', type=str, choices=['standard', 'class', 'lre'])
    parser.add_argument('--seed', '-s', type=int, default=1701,
                        help='Random seed of imbalanced mnist.')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=1e-2).setup(model)

    # Load the MNIST dataset
    train, val, test = get_imbalanced_mnist(seed=args.seed)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, len(val))
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    if args.update_algo == 'lre':
        updater = LRE_Updator({'main': train_iter, 'val': val_iter},
                              optimizer, device=args.gpu)
    elif args.update_algo == 'class':
        updater = ClassWeightUpdator(train_iter, optimizer, device=args.gpu)
    elif args.update_algo == 'standard':
        updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
