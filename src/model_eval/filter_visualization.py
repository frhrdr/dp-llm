# __author__ = 'frederik harder'
import matplotlib
matplotlib.use('Agg')  # to plot without Xserver
import matplotlib.pyplot as plt
import os
import shutil
import subprocess
from collections import namedtuple
import numpy as np
import torch as pt
import torch.nn as nn
from utils import save_img, transform_img_vals, get_data_loaders
from llm_mnist_model import LocallyLinearMnistModel


def get_mnist_settings():
    shared = '--lr 1e-3 --n-preds 30 -temp 30. -bs 500 -ep 20 --lr-decay-rate 0.8 --lr-decay-freq 5'
    shared_temp = 30.  # for loading models
    # 4 nonprivate models
    nonp_base = '{}'
    nonp_options = [' -rf 100', ' -rf 300', ' -rf 500', '']
    nonp_ids = ['epsx_rf100', 'epsx_rf300', 'epsx_rf500', 'epsx_rfx']
    nonp_strings = [shared + nonp_base.format(k) for k in nonp_options]

    # 6 private models
    dp_base = ' --sigma {} --max-norm 1e-3{}'
    dp_options = [(4.1, ''), (2.2, ''), (1.3, ''), (4.1, ' -rf 300'), (2.2, ' -rf 300'), (1.3, ' -rf 300')]
    dp_ids = ['eps05_rfx', 'eps1_rfx', 'eps2_rfx', 'eps05_rf300', 'eps1_rf300', 'eps2_rf300']
    dp_strings = [shared + dp_base.format(*k) for k in dp_options]

    run_ids = nonp_ids + dp_ids
    run_args = nonp_strings + dp_strings
    return run_ids, run_args, shared_temp


def get_fashion_settings():  # for attribution plots we only care about nonprivate models for now
    args_string = ['-bs 500 -ep 20 -lr 1e-3 --n-preds 30 --lr-decay-rate 0.8 --lr-decay-freq 5 --fashion']
    hi_dp_ids = ['epsx_rfx_fashion']
    shared_temp = 1.
    hi_dp_strings = args_string
    return hi_dp_ids, hi_dp_strings, shared_temp


def execute_runs(run_ids, run_args, base_save_dir, runs_offset=0):
    """
    trains and saves the specified setups to extract features later
    """
    base_cmd = ['python3.6', 'llm_mnist_model.py']

    for idx, run_id in enumerate(run_ids):
        run_arg_string = run_args[idx]
        if idx < runs_offset:  # for skipping parts of failed runs
            continue

        save_dir = base_save_dir.format(run_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        args_list = run_arg_string.split()
        added_args = ['--save-dir', save_dir]
        call_argv_list = base_cmd + args_list + added_args
        argv_string = ' '.join(call_argv_list)
        print('running:', argv_string)
        run_log = subprocess.run(call_argv_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            run_log.check_returncode()
        except subprocess.CalledProcessError:
            print(run_log.stderr.decode('UTF-8'))
            raise RuntimeError

    print('successfully ran all settings')


def prep_run(run_id, imgs_per_class, device, shared_temp, fashion, base_save_dir, n_labels, im_hw):
    run_save_dir = base_save_dir.format(run_id)
    model = LocallyLinearMnistModel.load_model(run_save_dir, temp=shared_temp).to(device)  # temp=None
    if model.rand_filt is not None:
        model.rand_filt = model.rand_filt.to(device)
    model.eval()

    img_mats, label_mats = load_testset_mats(fashion)
    img_choices = [img_mats[label_mats == l][:imgs_per_class] for l in range(n_labels)]
    img_choices = np.stack(img_choices).reshape(n_labels, imgs_per_class, 1, im_hw, im_hw)

    fake_batch = nn.Parameter(pt.Tensor(1, 1, im_hw, im_hw)).to(device)
    return model, img_choices, fake_batch, run_save_dir


def extract_from_img(img_choices, l, m, device, fake_batch, model, im_hw, run_save_dir, sub_dir):
    img_choice = img_choices[l, m:m + 1, :, :, :]
    fake_batch.data = pt.tensor(img_choice).to(device)

    _, sm_scores = model.forward(fake_batch, ret_softmax=True)  # (bs, k, np)
    sm_scores_l = sm_scores[0, l, :].detach().cpu().numpy()  # (np)

    if model.rand_filt is None:
        weights_l = model.weight[:-1, l, :]  # (self.d_in + 1, self.k, self.n_preds)  -> (d_in, np)
        weights_l = weights_l.detach().cpu().numpy()
    else:
        weights_l = model.weight[:, :-1, l]  # (np, rf + 1, k)  -> (np, rf)
        rand_filt = model.rand_filt  # (d_in, np, rf)
        effective_weights = pt.matmul(rand_filt.permute(1, 0, 2), weights_l[:, :, None])  # -> (np, d_in)
        weights_l = effective_weights.squeeze(2).permute(1, 0).detach().cpu().numpy()  # -> (d_in, np)

    assert len(sm_scores_l.shape) == 1
    order_ids = np.argsort(sm_scores_l)
    assert sm_scores_l[order_ids[-1]] == np.max(sm_scores_l)  # ascending sort

    sorted_scores_l = sm_scores_l[order_ids][::-1]  # descending sort
    sorted_w_l = weights_l[:, order_ids][:, ::-1]

    # now save the input image, the highest weight filters, and the associated weights
    img_lm = transform_img_vals(img_choice.reshape(im_hw, im_hw), pretty=False)

    img_save_dir = os.path.join(run_save_dir, '{}/l{}/m{}'.format(sub_dir, l, m))
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    save_img('{}/input.png'.format(img_save_dir), img_lm)

    return sorted_scores_l, sorted_w_l, img_save_dir


def load_and_plot(run_ids, base_save_dir, shared_temp, im_hw=28, use_cuda=True, n_labels=10, fashion=False):
    device = pt.device("cuda" if use_cuda else "cpu")
    imgs_per_class = 10
    top_n_filters = [1, 3, 5, 10, 30]

    for run_id in run_ids:
        model, img_choices, fake_batch, run_save_dir = prep_run(run_id, imgs_per_class, device,
                                                                shared_temp, fashion, base_save_dir,
                                                                n_labels, im_hw)

        for l in range(n_labels):
            for m in range(imgs_per_class):
                sub_dir = 'top_n_filters'
                sorted_scores_l, sorted_w_l, img_save_dir = extract_from_img(img_choices, l, m, device,
                                                                             fake_batch, model, im_hw, run_save_dir, sub_dir)

                for top_n in top_n_filters:
                    select_scores_l = sorted_scores_l[:top_n]
                    select_w_l = sorted_w_l[:, :top_n]

                    weighted_top_n = np.sum(select_scores_l[None, :] * select_w_l, axis=1)  # (d_in)
                    weighted_top_n = transform_img_vals(weighted_top_n.reshape(im_hw, im_hw))

                    save_img('{}/top_{}_filters.png'.format(img_save_dir, top_n), weighted_top_n)


def load_testset_mats(fashion):
    f_suffix = '_fashion' if fashion else ''
    return np.load('data/img_mats{}.npy'.format(f_suffix)), np.load('data/label_mats{}.npy'.format(f_suffix))


def extract_test_set_data(fashion=False):
    f_suffix = '_fashion' if fashion else ''
    bs = 100
    pseudo_args = namedtuple('args', ['batch_size', 'test_batch_size'])(bs, bs)
    _, test_loader = get_data_loaders(False, pseudo_args, fashion=fashion)
    label_mats = []
    img_mats = []
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data, labels
        label_mats.append(labels.numpy())
        img_mats.append(data.numpy())
    np.save('label_mats{}.npy'.format(f_suffix), np.concatenate(label_mats))
    np.save('img_mats{}.npy'.format(f_suffix), np.concatenate(img_mats))


def plot_filters_and_weights(run_ids, base_save_dir, shared_temp, im_hw=28, use_cuda=True, n_labels=10, fashion=False):
    device = pt.device("cuda" if use_cuda else "cpu")
    imgs_per_class = 30
    top_n_filters = 5

    for run_id in run_ids:
        model, img_choices, fake_batch, run_save_dir = prep_run(run_id, imgs_per_class, device,
                                                                shared_temp, fashion, base_save_dir,
                                                                n_labels, im_hw)

        for l in range(n_labels):
            for m in range(imgs_per_class):

                sub_dir = 'in_class_filter_comp'
                sorted_scores_l, sorted_w_l, img_save_dir = extract_from_img(img_choices, l, m, device,
                                                                             fake_batch, model, im_hw,
                                                                             run_save_dir, sub_dir)

                np.save('{}/softmax_weights.npy'.format(img_save_dir), sorted_scores_l)
                with open('{}/softmax_weights.txt'.format(img_save_dir), mode='w') as f:
                    f.writelines(['{0:.10f}\n'.format(sorted_scores_l[k]) for k in range(sorted_scores_l.shape[0])])

                for n in range(1, top_n_filters + 1):
                    select_w_l = sorted_w_l[:, n]
                    select_w_l = transform_img_vals(select_w_l.reshape(im_hw, im_hw))

                    save_img('{}/top_{}_filter.png'.format(img_save_dir, n), select_w_l)


def aggregate_filters_for_class(base_save_dir, run_id, label, n_images=30, top_n=5):
    img_load_dir = os.path.join(base_save_dir.format(run_id), 'in_class_filter_comp/l{}'.format(label))

    softmax_weights_acc = []
    final_grid_acc = []
    for m in range(n_images):
        m_dir = os.path.join(img_load_dir, 'm{}/'.format(m))
        img = plt.imread(os.path.join(m_dir, 'input.png'))
        filts = [plt.imread(os.path.join(m_dir, 'top_{}_filter.png'.format(k))) for k in range(1, top_n+1)]
        row = np.concatenate([img] + filts, axis=1)
        final_grid_acc.append(row)
        softmax_weights_acc.append(np.load(os.path.join(m_dir, 'softmax_weights.npy'))[:top_n])

    final_grid_acc = np.concatenate(final_grid_acc, axis=0)
    save_img(os.path.join(img_load_dir, 'grid_aggregate.png'), final_grid_acc)
    softmax_weights_acc = np.stack(softmax_weights_acc)
    np.save(os.path.join(img_load_dir, 'sm_weights.npy'), softmax_weights_acc)
    print('softmax weights:')
    for m in range(n_images):
        print(softmax_weights_acc[m, :].T)


def get_all_imgs():
    base_save_dir = 'filter_vis_data/{}/'
    run_ids, run_args, shared_temp = get_mnist_settings()
    execute_runs(run_ids, run_args, base_save_dir, runs_offset=0)
    load_and_plot(run_ids, base_save_dir, shared_temp)


def collect_for_eps_plot(label, img_id, top_n):
    src_dirs = ['epsx_rfx', 'eps2_rfx', 'eps1_rfx', 'eps05_rfx']
    vis_name = 'epsilon_high_dp'
    collect_imgs_for_plot(label, img_id, top_n, src_dirs, vis_name)


def collect_for_rf_plot(label, img_id, top_n):
    src_dirs = ['epsx_rfx', 'epsx_rf500', 'epsx_rf300', 'epsx_rf100']
    vis_name = 'rand_filt'
    collect_imgs_for_plot(label, img_id, top_n, src_dirs, vis_name)


def collect_for_rf_eps_plot(label, img_id, top_n):
    src_dirs = ['epsx_rf300', 'eps2_rf300', 'eps1_rf300', 'eps05_rf300']
    vis_name = 'epsilon_rf_high_dp'
    collect_imgs_for_plot(label, img_id, top_n, src_dirs, vis_name)


def collect_imgs_for_plot(label, img_id, top_n, src_dirs, vis_name):
    base_load_str = 'filter_vis_data/{}/img_l{}_m{}/{}.png'
    img_name = 'input'
    top_name = 'top_1_filters'
    avg_name = 'top_{}_filters'.format(top_n)

    base_save_dir = 'filter_vis/{}/select_l{}_m{}_top{}'.format(vis_name, label, img_id, top_n)
    tgt_dirs = ['base', 'easy', 'medium', 'hard']
    for src_dir, tgt_dir in zip(src_dirs, tgt_dirs):
        tgt_save_dir = os.path.join(base_save_dir, tgt_dir)
        if not os.path.exists(tgt_save_dir):
            os.makedirs(tgt_save_dir)
        shutil.copy(base_load_str.format(src_dir, label, img_id, top_name), os.path.join(tgt_save_dir, 'top.png'))
        shutil.copy(base_load_str.format(src_dir, label, img_id, avg_name), os.path.join(tgt_save_dir, 'avg.png'))

    shutil.copy(base_load_str.format(src_dirs[0], label, img_id, img_name), os.path.join(base_save_dir, 'input.png'))


def make_collections(labels, img_ids, top_ns):
    """
    sorts images used for fig.4 in the paper into subfolders
    """
    for l in labels:
        for m in img_ids:
            for top_n in top_ns:

                collect_for_rf_plot(l, m, top_n)
                collect_for_rf_eps_plot(l, m, top_n)
                collect_for_eps_plot(l, m, top_n)


def main():
    """
    use this script to train a number of LLMs as described in the various settings with 'execute_runs' and then
    'load_and_plot' the highest activated filters (and their weighted sums) belonging to the target class given
    a number of test set images.
    """
    b_save_dir = 'filter_vis_data/{}/'
    fashion = True
    run_ids, run_args, shared_temp = get_mnist_settings()
    execute_runs(run_ids, run_args, b_save_dir, runs_offset=0)
    load_and_plot(run_ids, b_save_dir, shared_temp, im_hw=28, use_cuda=True, n_labels=10, fashion=fashion)
    plot_filters_and_weights(run_ids, b_save_dir, shared_temp, im_hw=28, use_cuda=True, n_labels=10, fashion=fashion)

    # for label in range(10):
    #     aggregate_filters_for_class(b_save_dir, run_id='epsx_rfx_fashion', label=label, n_images=30, top_n=5)
    # make_collections(list(range(10)), [2, 5], [30])


if __name__ == '__main__':
    main()
