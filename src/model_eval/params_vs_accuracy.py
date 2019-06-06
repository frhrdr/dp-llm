# __author__ = 'frederik harder'
import os
import numpy as np
import subprocess
import argparse


def get_save_path_extension(setting_id, seed_id):
    return 'runs/setting_{}/seed_{}/'.format(setting_id, seed_id)


def execute_runs(base_save_dir, base_args, settings, seeds, setting_offset=0, seed_offset=0):
    base_cmd = ['python3.6', 'llm_mnist_model.py']

    for idx, setting in enumerate(settings):
        if idx < setting_offset:  # for skipping parts of failed runs
            continue
        args_list = base_args.format(*setting).split()
        for idy, seed in enumerate(seeds):
            if idx == setting_offset and idy < seed_offset:
                continue
            save_dir = os.path.join(base_save_dir, get_save_path_extension(idx, idy))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            added_args = ['--seed', str(seed), '--setup', save_dir]
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


def aggregate_logged_test_accuracies(base_save_dir, n_settings, n_seeds, last_n_epochs_only=None):
    all_accs = []
    for idx in range(n_settings):
        setting_accs = []
        for idy in range(n_seeds):
            save_file = os.path.join(base_save_dir, get_save_path_extension(idx, idy), 'test_accuracies.npy')
            acc_vec = np.load(save_file)
            if last_n_epochs_only is not None:
                acc_vec = acc_vec[-last_n_epochs_only:]
                print(acc_vec.shape)
            setting_accs.append(acc_vec)  # (epochs) vector
        all_accs.append(np.stack(setting_accs, axis=0))

    all_accs = np.stack(all_accs, axis=0)
    np.save(os.path.join(base_save_dir, 'test_accuracies.npy'), all_accs)  # (n_settings, n_seeds, n_epochs)

    means = np.mean(all_accs, axis=1)  # (n_settings, n_epochs)
    np.save(os.path.join(base_save_dir, 'means.npy'), means)

    sdevs = np.std(all_accs, axis=1)  # (n_settings, n_epochs)
    print('sdevs-shape', sdevs.shape)
    np.save(os.path.join(base_save_dir, 'sdevs.npy'), sdevs)
    print('collected all test logs as numpy arrays')


def numpy_to_text_file(base_save_dir, x_vals):
    all_accs = np.load(os.path.join(base_save_dir, 'test_accuracies.npy'))
    means = np.load(os.path.join(base_save_dir, 'means.npy'))
    sdevs = np.load(os.path.join(base_save_dir, 'sdevs.npy'))

    # only care about final output atm.
    all_accs = all_accs[:, :, -1]  # (n_settings, n_seeds)
    means = means[:, -1]  # (n_settings)
    sdevs = sdevs[:, -1]  # (n_settings)
    n_settings, n_seeds = all_accs.shape

    header = ' '.join(['x_val'] + ['acc{}'.format(k) for k in range(n_seeds)] + ['acc_mean', 'acc_std'])
    text = [header]
    for idx in range(n_settings):
        line = ' '.join([str(x_vals[idx])] +
                        [str(all_accs[idx, k]) for k in range(n_seeds)] +
                        [str(means[idx]), str(sdevs[idx])])
        text.append(line)

    print(text)
    file_name = os.path.join(base_save_dir, 'text_log.txt')

    with open(file_name, mode='w') as f:
        f.write('\n'.join(text))


def get_epsilon_run_params():
    base_save_dir = 'plot_data/epsilon_plot'
    base_args = '-bs {} -ep {} --sigma {} --n-preds 30 --lr 1e-3 -rf 300 --max-norm 1e-3 ' \
                '--lr-decay-rate {} --lr-decay-freq {} -temp 30.'
    settings = [(500, 20, 4.1, 0.8, 5), (500, 20, 2.2, 0.8, 5), (500, 20, 1.3, 0.8, 5), (1500, 60, 1.8, 0.9, 10),
                (1500, 60, 1.1, 0.9, 1)]
    x_vals = [0.5, 1, 2, 4, 8]  # x axis of plot: epsilons
    return base_save_dir, base_args, settings, x_vals


def get_epsilon_fashion_run_params():
    base_save_dir = 'plot_data/epsilon_fashion_plot'
    base_args = '-bs {} -ep {} --sigma {} --n-preds 30 --lr 1e-3 -rf 300 --max-norm 1e-3 ' \
                '--lr-decay-rate {} --lr-decay-freq {} --fashion'
    settings = [(500, 20, 4.1, 0.8, 5), (500, 20, 2.2, 0.8, 5), (500, 20, 1.3, 0.8, 5), (1500, 60, 1.8, 0.9, 10),
                (1500, 60, 1.1, 0.9, 1)]
    x_vals = [0.5, 1, 2, 4, 8]  # x axis of plot: epsilons
    return base_save_dir, base_args, settings, x_vals


def get_n_preds_non_dp_run_params():
    base_save_dir = 'plot_data/n_preds_plot'
    base_args = '-bs 500 -ep 20 --n-preds {} --lr 1e-3 -rf 300 --lr-decay-rate 0.8 --lr-decay-freq 5 -temp 30. '
    settings = [(1,), (5,), (10,), (30,), (100,)]
    x_vals = [1, 5, 10, 30, 100]  # x axis of plot: n_preds
    return base_save_dir, base_args, settings, x_vals


def get_n_preds_dp_run_params():
    base_save_dir = 'plot_data/n_preds_dp_plot'
    base_args = '-bs 500 -ep 20 --n-preds {} --lr 1e-3 -rf 300 --lr-decay-rate 0.8 --lr-decay-freq 5 -temp 30. ' \
                '--sigma 1.3 --max-norm 1e-3'
    settings = [(1,), (5,), (10,), (30,), (100,)]
    x_vals = [1, 5, 10, 30, 100]  # x axis of plot: n_preds
    return base_save_dir, base_args, settings, x_vals


def get_n_rand_filters_non_dp_run_params():
    base_save_dir = 'plot_data/rand_filters_plot'
    base_args = '-bs 500 -ep 20 --n-preds 30 --lr 1e-3 {}--lr-decay-rate 0.8 --lr-decay-freq 5 -temp 30.'
    settings = [('-rf {} '.format(k),) for k in (50, 100, 300, 500)] + [('',)]  # also no projection case
    x_vals = [50, 100, 300, 500, 'direct input']  # x axis of plot: number of random filters
    return base_save_dir, base_args, settings, x_vals


def get_n_rand_filters_dp_run_params():
    base_save_dir = 'plot_data/rand_filters_dp_plot'
    base_args = '-bs 500 -ep 20 --n-preds 30 --lr 1e-3 {}--lr-decay-rate 0.8 --lr-decay-freq 5 -temp 30. ' \
                '--sigma 1.3 --max-norm 1e-3'
    settings = [('-rf {} '.format(k),) for k in (50, 100, 300, 500)] + [('',)]  # also no projection case
    x_vals = [50, 100, 300, 500, 'direct input']  # x axis of plot: number of random filters
    return base_save_dir, base_args, settings, x_vals


def get_exp_params(key):
    if key == 'epsilon':
        return get_epsilon_run_params()
    elif key == 'epsilon-fashion':
        return get_epsilon_fashion_run_params()
    elif key == 'np':
        return get_n_preds_non_dp_run_params()
    elif key == 'np-dp':
        return get_n_preds_dp_run_params()
    elif key == 'rf':
        return get_n_rand_filters_non_dp_run_params()
    elif key == 'rf-dp':
        return get_n_rand_filters_dp_run_params()
    else:
        raise ValueError


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--setting-offset', type=int, default=0)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--last-n-epochs-only', type=int, default=None)
    parser.add_argument('--no-runs', action='store_true', default=False)
    args = parser.parse_args()

    base_save_dir, base_args, settings, x_vals = get_exp_params(args.exp)
    seeds = [24, 221, 53, 64, 4562, 34, 23, 73, 3, 532]
    if not args.no_runs:
        execute_runs(base_save_dir, base_args, settings, seeds,
                     setting_offset=args.setting_offset, seed_offset=args.seed_offset)
    aggregate_logged_test_accuracies(base_save_dir, len(settings), len(seeds), args.last_n_epochs_only)
    numpy_to_text_file(base_save_dir, x_vals)


if __name__ == '__main__':
    main()
