# __author__ = 'frederik harder'
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import numpy as np
import torch as pt
from torchvision import transforms, datasets


def get_data_loaders(use_cuda, args, fashion=False, shuffle=True, ret_data=False):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if fashion:
        fmnist_mean = 0.
        fmnist_std = 1.
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((fmnist_mean,), (fmnist_std,))])
        trn_data = datasets.FashionMNIST('../data/fmnist', train=True, download=True, transform=transform)
        tst_data = datasets.FashionMNIST('../data/fmnist', train=False, transform=transform)
        train_loader = pt.utils.data.DataLoader(trn_data, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        test_loader = pt.utils.data.DataLoader(tst_data, batch_size=args.test_batch_size, shuffle=shuffle, **kwargs)
    else:
        mnist_mean = 0.1307
        mnist_std = 0.3081
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_std,))])
        trn_data = datasets.MNIST('../data/mnist', train=True, download=True, transform=transform)
        tst_data = datasets.MNIST('../data/mnist', train=False, transform=transform)
        train_loader = pt.utils.data.DataLoader(trn_data, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
        test_loader = pt.utils.data.DataLoader(tst_data, batch_size=args.test_batch_size, shuffle=shuffle, **kwargs)

    if ret_data:
        return train_loader, test_loader, trn_data, tst_data
    else:
        return train_loader, test_loader


def rand_orthonormal(d_in, n_preds, d_out):
    rand_mats = [np.random.normal(np.zeros((d_in, d_out))) for _ in range(n_preds)]
    orth_mats = [scipy.linalg.orth(k) for k in rand_mats]
    for m in orth_mats:
        assert m.shape[1] == d_out
    orthonorm_mats = [k / np.tile(np.linalg.norm(k, axis=1, keepdims=True), (1, d_out)) for k in orth_mats]
    mat = np.stack(orthonorm_mats, axis=1).astype(np.float32)
    return np.reshape(mat, (d_in, n_preds * d_out))


def save_img(save_file, img):
    plt.imsave(save_file, img, cmap=cm.gray, vmin=0., vmax=1.)


def transform_img_vals(x, pretty=False):
    if pretty:
        x = x - x[0]
        return x / (2. * np.max(np.abs(x))) + 0.5
    else:
        x = x - np.min(x)
        return x / np.max(x)