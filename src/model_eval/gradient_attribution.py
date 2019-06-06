# __author__ = 'frederik harder'
import numpy as np
import torch as pt
import torch.nn as nn
import os
from llm_mnist_model import LocallyLinearMnistModel
from mnist_cnn import MnistCNN
from utils import save_img, transform_img_vals


def load_mnist_model(device, fashion, cnn):
    if cnn:
        model = MnistCNN().to(device)
        load_path = 'saved-models/baseline-cnns/mnist_cnn{}.pt'.format('_fashion' if fashion else '')
        model.load_state_dict(pt.load(load_path))
    else:
        model = LocallyLinearMnistModel.load_model('saved-models/logreg-model/nonDP-300-preds-mnist').to(device)
        if model.rand_filt is not None:
            model.rand_filt = model.rand_filt.to(device)
    return model


def get_attributions(labels_to_plot, cnn, fashion, batch_fun, batch_size, imgs_per_class, im_hw, use_cuda):
    f_suffix = '_fashion' if fashion else ''
    device = pt.device("cuda" if use_cuda else "cpu")
    model = load_mnist_model(device, fashion=False, cnn=cnn)

    save_dir = 'plots_feat_grad_vis/for-comparison{}/{}-plots'.format(f_suffix, 'cnn' if cnn else 'lin')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.eval()
    label_mats = np.load('label_mats{}.npy'.format(f_suffix))
    img_mats = np.load('img_mats{}.npy'.format(f_suffix))
    fake_batch = nn.Parameter(pt.Tensor(batch_size, 1, im_hw, im_hw)).to(device)

    imgs = []
    grads = []
    for l in labels_to_plot:
        imgs_l = []
        grads_l = []
        ids_l = label_mats == l
        for m in range(imgs_per_class):
            img_lm = img_mats[ids_l][m].reshape(1, im_hw, im_hw)

            img_batch = batch_fun(img_lm)  # different for integrated and smooth grad method
            fake_batch.data = pt.tensor(img_batch).to(device)
            fake_batch.retain_grad()

            fake_batch.grad = pt.zeros(batch_size, 1, 28, 28).to(device)
            model_out = model(fake_batch)
            loss = pt.sum(model_out[:, l])
            loss.backward()

            grad_lm = pt.mean(fake_batch.grad, dim=0)

            grad_lm = transform_img_vals(grad_lm.cpu().numpy())
            img_lm = transform_img_vals(img_lm)

            imgs_l.append(img_lm.reshape((28, 28)))
            grads_l.append(grad_lm.reshape((28, 28)))

        grads.append(np.concatenate(grads_l, axis=0))
        imgs.append(np.concatenate(imgs_l, axis=0))

    grad_plot = np.concatenate(grads, axis=1)
    img_plot = np.concatenate(imgs, axis=1)
    return grad_plot, img_plot, save_dir


def plot_integrated_grads(labels_to_plot, cnn, fashion, n_intervals=600, imgs_per_class=10, im_hw=28, use_cuda=True):
    def int_grad_batch_fun(img):
        return np.stack([(k / n_intervals) * img for k in range(1, n_intervals + 1)])
    grad_plot, img_plot, save_dir = get_attributions(labels_to_plot, cnn, fashion, int_grad_batch_fun,
                                                     batch_size=n_intervals,
                                                     imgs_per_class=imgs_per_class, im_hw=im_hw, use_cuda=use_cuda)
    save_img('{}/int_grads_{}_steps.png'.format(save_dir, n_intervals), grad_plot)
    save_img('{}/int_grad_imgs.png'.format(save_dir), img_plot)


def plot_smooth_grads(labels_to_plot, cnn, fashion, n_samples=600, imgs_per_class=10, noise_level=0.1, im_hw=28,
                      use_cuda=True):
    def smooth_grad_batch_fun(img):
        noise_sigma = noise_level * (np.max(img) - np.min(img))
        return np.stack([img + np.random.normal(np.zeros_like(img), noise_sigma)
                         for _ in range(n_samples)]).astype(np.float32)
    grad_plot, img_plot, save_dir = get_attributions(labels_to_plot, cnn, fashion, smooth_grad_batch_fun,
                                                     batch_size=n_samples,
                                                     imgs_per_class=imgs_per_class, im_hw=im_hw, use_cuda=use_cuda)
    save_img('{}/smooth_grad_{}_noise_{}_samples.png'.format(save_dir, noise_level, n_samples), grad_plot)
    save_img('{}/smooth_grad_imgs.png'.format(save_dir), img_plot)


if __name__ == '__main__':
    pass
