# __author__ = 'mijung'
import numpy as np
import moments_accountant as ma


def calculate_epsilon(n_data=60000, n_epochs=20, batch_size=50, sigma=1., total_del=1e-5, verify=True):
    """ ========  calculate  moments  ======== """

    steps = n_data / batch_size * n_epochs  # number of iterations
    q = (batch_size / n_data)  # sampling rate

    # make sure lambda < sigma^2 log (1/(nu*sigma))
    max_lmbd = int(np.floor((sigma ** 2) * np.log(1 / (q * sigma))))
    max_lmbd *= 2  # increase val for good measure

    lmbds = range(1, max_lmbd + 1)
    log_moments = []
    for lmbd in lmbds:
        # log_moment = 0
        log_moment = ma.compute_log_moment(q, sigma, steps, lmbd, verify=verify, verbose=False)
        # print(log_moment)
        log_moments.append((lmbd, log_moment))

    total_epsilon, total_delta = ma.get_privacy_spent(log_moments, target_delta=total_del)

    print("total privacy loss computed by moments accountant is {}".format(total_epsilon))

    return total_epsilon, total_delta


def main():
    n_data = 60000
    sigmas = [1.1, 1.2, 1.7]
    epochs = [60]
    batch_size = 1500
    ma_delta = 1e-5
    epsilons = []
    for s in sigmas:
        for ep in epochs:
            eps, delta = calculate_epsilon(n_data, ep, batch_size, s, total_del=ma_delta, verify=False)
            epsilons.append(eps)
            print('sigma:', s, 'delta:', ma_delta)
            print('n_epochs: ', ep)
            print('')
            print('-----------------------------------------------------------------------------------------------')
    print('epsilons', epsilons)


if __name__ == '__main__':
    main()
