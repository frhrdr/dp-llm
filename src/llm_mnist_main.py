# __author__ = 'frederik harder'
import numpy as np
import torch as pt
import torch.nn.functional as nnf
import os
from collections import namedtuple
import argparse
from utils import get_data_loaders
from llm_mnist_model import LocallyLinearMnistModel


def main(args):

    if args.seed is not None:
        np.random.seed(args.seed)
        pt.manual_seed(args.seed)

    pseudo_args = namedtuple('args', ['batch_size', 'test_batch_size'])(args.batch_size, args.batch_size)
    train_loader, test_loader, train_data, test_data = get_data_loaders(use_cuda=True, args=pseudo_args,
                                                                        fashion=args.fashion, ret_data=True)

    model = LocallyLinearMnistModel(args.n_preds, args.d_rand_filt, no_bias=args.no_bias, temp=args.softmax_temp
                                    ).to(args.device)
    if model.rand_filt is not None:
        model.rand_filt = model.rand_filt.to(args.device)

    optimizer = pt.optim.Adam(model.parameters(), args.lr)

    test_accs = []
    for epoch in range(1, args.epochs + 1):
        train(args, train_loader, optimizer, model, epoch)
        acc = test(args, test_loader, model)
        test_accs.append(acc)

        if args.lr_decay_freq is not None and epoch % args.lr_decay_freq == 0:
            args.lr = args.lr * args.lr_decay_rate
            print('new lr: ', args.lr)
            optimizer = pt.optim.Adam(model.parameters(), args.lr)

    if args.setup is not None:
        log_accuracy(test_accs, args.setup)

    if args.save_dir:
        model.save_model(args.save_dir)


def log_accuracy(test_acc, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    np.save(os.path.join(log_dir, 'test_accuracies.npy'), np.asarray(test_acc))


def train(args, train_loader, optimizer, model, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        loss = nnf.nll_loss(model(data), target)
        model.private_update(loss, optimizer, args.max_norm, args.sigma)

        if batch_idx % 60 == 0:
            print('Train Ep: {} {:.0f}%\tLoss: {:.6f}'.format(epoch, 100. * batch_idx / len(train_loader), loss.item()))


def test(args, test_loader, model):
    model.eval()
    correct = 0
    with pt.no_grad():
        for data, target in test_loader:
            pred = pt.argmax(model(data.to(args.device)), dim=-1)
            correct += pred.eq(target.to(args.device).view_as(pred)).sum().item()

    acc = correct / len(test_loader.dataset)
    print('\nTest Accuracy: {:.2f}%\n'.format(100. * acc))

    return acc


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda')  # or cpu
    parser.add_argument('--batch-size', '-bs', type=int, default=100)
    parser.add_argument('--n-labels', type=int, default=10)  # 10 classes on mnist
    parser.add_argument('--epochs', '-ep', type=int, default=10)

    parser.add_argument('--lr', '-lr', type=float, default=0.001)
    parser.add_argument('--lr-decay-rate', type=float, default=None)  # multiplies current lr with this factor...
    parser.add_argument('--lr-decay-freq', type=int, default=None)  # ... after every nth epoch

    parser.add_argument('--n-preds', '-np', type=int, default=3)  # number of predictions
    parser.add_argument('--d-rand-filt', '-rf', type=int, default=None)  # out dim of random filters
    parser.add_argument('--softmax-temp', '-temp', type=float, default=1.)

    parser.add_argument('--fashion', action='store_true', default=False)
    parser.add_argument('--no-bias', action='store_true', default=False)

    parser.add_argument('--sigma', '-sig', type=float, default=None)  # privacy parameter
    parser.add_argument('--max-norm', '-clip', type=float, default=None)  # per sample gradient norm clipping threshold

    parser.add_argument('--setup', type=str, default='multi_lin_test')  # run id for logging
    parser.add_argument('--save-dir', type=str, default=None)

    parser.add_argument('--seed', '-seed', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    line_args = parse_args()
    main(line_args)
