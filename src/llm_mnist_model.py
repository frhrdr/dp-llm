# __author__ = 'frederik harder'
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import os
import math
from utils import rand_orthonormal


class LocallyLinearMnistModel(nn.Module):
    def __init__(self, n_preds, d_rand_filt, no_bias=False, orth_projection=False, temp=None):
        super(LocallyLinearMnistModel, self).__init__()
        self.d_in = 784
        self.d_rand_filt = d_rand_filt
        self.k = 10
        self.rf_scale = 1 / np.sqrt(2)  # scaling hyperparameter
        self.sm_temp = temp
        self.n_preds = n_preds
        self.h, self.z = None, None
        self.no_bias = no_bias
        if d_rand_filt is None:
            self.rand_filt = None
            self.weight = nn.Parameter(pt.Tensor(self.d_in + 1, self.k, self.n_preds))
        else:
            if orth_projection:
                rf_mat = rand_orthonormal(self.d_in, self.n_preds, self.d_rand_filt) * self.rf_scale
            else:
                rf_mat = np.random.normal(loc=np.zeros((self.d_in, self.n_preds, self.d_rand_filt))) * self.rf_scale
            self.rand_filt = pt.tensor(rf_mat.astype(np.float32))
            self.weight = nn.Parameter(pt.Tensor(self.n_preds, self.d_rand_filt + 1, self.k))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_bias(self, size, device):
        if self.no_bias:
            return pt.zeros(size, dtype=pt.float32, device=device)
        else:
            return pt.ones(size, dtype=pt.float32, device=device)

    def forward(self, x, ret_softmax=False):  # (bs, din) --> (bs, k, np, dout)
        bs = x.size()[0]
        if self.d_rand_filt is None:
            bias_units = self.get_bias((bs, 1), x.get_device())
            self.h = pt.cat([x.view(-1, self.d_in), bias_units], dim=-1)  # add bias
            z_flat = pt.matmul(self.h, self.weight.view(self.d_in + 1, self.k * self.n_preds))
            self.z = z_flat.view(bs, self.k, self.n_preds)
            self.z.retain_grad()
            z_softmax = nnf.softmax(self.z, dim=2)  # (bs, k, np)
            per_class_preds = pt.sum(z_softmax * self.z, dim=2)  # (bs, k, np) -> (bs, k)
        else:
            # (bs, d_in) x (d_in, np, rf) -> (bs, np * rf)
            rf_flat = self.rand_filt.view(self.d_in, self.n_preds * self.d_rand_filt)
            rand_proj = pt.matmul(x.view(-1, self.d_in), rf_flat)
            bias_units = self.get_bias((bs, self.n_preds, 1), x.get_device())
            self.h = pt.cat([rand_proj.view(bs, self.n_preds, self.d_rand_filt), bias_units], dim=-1)  # add bias
            # (bs, np, 1, rf+1) x (np, rf+1, k) -> (bs, np, k)
            self.z = pt.matmul(self.h[:, :, None, :], self.weight).squeeze(2)  # (bs, np, k)
            self.z.retain_grad()
            z_softmax = nnf.softmax(self.z / self.sm_temp, dim=1)  # (bs, np, k)
            per_class_preds = pt.sum(z_softmax * self.z, dim=1)  # (bs, np, k) -> (bs, k)
            if ret_softmax:
                z_softmax = z_softmax.permute(0, 2, 1)  # (bs, k, np)

        log_sm = nnf.log_softmax(per_class_preds, dim=-1)
        return log_sm if not ret_softmax else (log_sm, z_softmax)

    def private_update(self, loss, optimizer, max_norm, sigma):
        loss.backward()
        per_sample_grads = self.compute_samplewise_grads()
        self.clip_samplewise_grads(per_sample_grads, max_norm)
        if sigma is not None:
            self.add_noise(sigma, max_norm)
        optimizer.step()

    def compute_samplewise_grads(self):
        if self.d_rand_filt is None:
            # normal - h:(bs, d_in+1)  z:(bs, k, np) -> (bs, d_in+1, k, np)
            z_view = self.z.grad.view(-1, 1, self.k * self.n_preds)
            return pt.matmul(self.h[:, :, None], z_view).view(-1, self.d_in + 1, self.k, self.n_preds)
        else:
            # projected - h:(bs, np, din+1)  z:(bs, np, k)  ->  (bs, np, d_in+1, k)
            return pt.matmul(self.h[:, :, :, None], self.z.grad[:, :, None, :])

    def clip_samplewise_grads(self, per_sample_grads, max_norm):  # calc global grad norm per sample, mult all by factor
        # per_sample grads is (bs, d_in+1, k, np)
        if max_norm is not None:
            sum_dims = [1, 2, 3]  # sum over all dims but 0
            grad_norms = pt.sqrt(pt.sum(per_sample_grads ** 2, dim=sum_dims))

            norm_factors = pt.min(max_norm / grad_norms, pt.ones_like(grad_norms))  # (bs)
            n_facs = norm_factors[:, None, None, None]
            per_sample_grads = per_sample_grads * n_facs
        w_grad = pt.sum(per_sample_grads, dim=0)

        self.weight.grad = w_grad

    def add_noise(self, sigma, max_norm):
        self.weight.grad += pt.normal(mean=pt.zeros_like(self.weight), std=sigma * max_norm)

    def save_model(self, save_dir):
        # save weights, random filters, means
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pt.save(self.state_dict(), os.path.join(save_dir, 'model_weights.pt'))
        if self.rand_filt is not None:
            pt.save(self.rand_filt, os.path.join(save_dir, 'model_random_filters.pt'))

    @staticmethod
    def load_model(load_dir, no_bias=False, orth_projection=False, temp=None):
        # load weights, random filters, means
        # infer n_preds, d_rand_filt
        s_dict = pt.load(os.path.join(load_dir, 'model_weights.pt'))
        w_size = s_dict['weight'].size()
        if w_size[0] == 785:  # no random filter
            assert w_size[1] == 10
            n_preds = w_size[2]  # w: self.d_in + 1, self.k, self.n_preds
            model = LocallyLinearMnistModel(n_preds, d_rand_filt=None,
                                            no_bias=no_bias, orth_projection=orth_projection, temp=temp)
            model.load_state_dict(s_dict)

        else:  # shared random filters
            n_preds, rf_bias, k = w_size  # w: self.n_preds, self.d_rand_filt + 1, self.k
            assert k == 10
            rand_filters = pt.load(os.path.join(load_dir, 'model_random_filters.pt'))
            model = LocallyLinearMnistModel(n_preds, d_rand_filt=rf_bias - 1,
                                            no_bias=no_bias, orth_projection=orth_projection, temp=temp)
            model.load_state_dict(s_dict)
            model.rand_filt = rand_filters

        return model
