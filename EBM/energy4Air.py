import math
import torch
from torch import nn
from typing import List
import torch.autograd as autograd
import numpy as np
import random

from .utils import config_activation, gaussian_repar, gaussian_log_prob, MLP
from .energy_tensor_cnce import GaussianFourierProjection
INIT_QZ_SIGMA = 0.2

class _ScoreTRBase(nn.Module):
    def __init__(
        self, tensor_shape, rank, h_dim, act, dropout,
        latent_dim, embedding_size, sigma_func, pooling_method,
        skip_connection):
        super(_ScoreTRBase, self).__init__()
        self.tensor_shape = list(tensor_shape)
        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.embedding_size = embedding_size
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection

        self.dim = len(tensor_shape)

        self.setup_q_z_()
        self.setup_energy_()

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])

    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [12, 5] [6, 5]
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            torch.nn.init.normal_(q.data, -2, INIT_QZ_SIGMA)

    def setup_energy_(self):
        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(
            input_dim=self.dim * self.rank, output_dim=self.latent_dim,
            h_dim=self.h_dim, act=act, dropout=self.dropout,
            bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # x encoder
        if hasattr(self, 'category'):
            if self.embedding_size is not None:
                emb_size = self.embedding_size
            else:
                emb_size = self.category
        else:
            emb_size = 1 if self.embedding_size == 1 else 2 * self.embedding_size
        x_enc = MLP(
            input_dim=emb_size, output_dim=self.latent_dim,
            h_dim=self.h_dim, act=act, dropout=self.dropout,
            bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # ouput layer
        if self.pooling_method in ['sum', 'attn']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 2 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(
            input_dim=in_size, output_dim=int(in_size / 2),
            h_dim=self.h_dim, act=act, dropout=self.dropout,
            bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        self.layers = nn.ModuleDict({
            'z_enc': z_enc, 'x_enc': x_enc, 'output': output_layer
        })

    def _input_embedding(self, x):
        raise NotImplementedError

    def forward(self, idx, x, sample=True, return_z=False):
        z = []
        for d in range(self.dim):
            if sample:
                z_d = gaussian_repar(
                    mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))
            else:
                z_d = self.q_z_mu[d]
            z.append(z_d[idx[:, d]])
        z_ten = torch.cat(z, -1)

        # expand input
        x_exp = self._input_embedding(x)
        x_exp = self.layers['x_enc'](x_exp)
        z_exp = self.layers['z_enc'](z_ten)

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([z_exp, x_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        else:
            raise RuntimeError('Wrong pooling method!')
        energy = self.layers['output'](hidden).pow(2).sum(-1)

        if return_z:
            return - energy, z
        else:
            return - energy

    def predict(self, idx=None, batch=1e4, sampling=False, x0=None,
        step=100, epsilon=0.02,  # learning rate or grid interval
        t=1., x_range=None):
        if idx is not None:
            idx_array = [idx]
            idx_list = None
            idx_full = None
        else:
            idx_array, idx_list, idx_full = self._split_idx(batch)

        tensor_predict = []
        chunk_num = len(idx_array)
        for n in range(chunk_num):
            if sampling:  # use Langevin Dynamics sampling
                if x0 is None:
                    x0_n = None
                else:
                    x0_n = x0[idx_list[n]]
                x_hat = self.langevin_sample_posterior(idx_array[n], x0=x0_n, step=step, epsilon=epsilon, t=t)
            else:  # use grid search for MAP
                x_hat = self.grid_search_posterior(idx=idx_array[n], x_range=x_range, epsilon=epsilon)

            tensor_predict.append(x_hat)

        tensor_predict = torch.cat(tensor_predict).squeeze()
        if idx is not None:
            return tensor_predict

        tensor_full = tensor_predict.view(*self.tensor_shape)
        return tensor_full

    def _split_idx(self, batch):
        device = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype

        tensor_full = torch.ones(self.tensor_shape, dtype=dtype, device=device)
        idx = torch.nonzero(tensor_full)
        if idx.shape[0] < batch:
            chunk_num = 1
        else:
            chunk_num = int(idx.shape[0] / batch)
        idx_array = torch.chunk(idx, chunk_num)

        idx_list_ = torch.nonzero(tensor_full, as_tuple=True)
        idx_list_ = [torch.chunk(i, chunk_num) for i in idx_list_]
        idx_list = []
        for i in range(chunk_num):
            idx_cache = []
            for d in range(tensor_full.ndim):
                idx_cache.append(idx_list_[d][i])
            idx_list.append(tuple(idx_cache))

        return idx_array, idx_list, idx_list_


class ScoreMatchingTime(_ScoreTRBase):
    def __init__(
        self,
        tensor_shape: List[int],
        rank: int,
        h_dim: List[int],
        act: str = 'elu',
        dropout: float = 0.,
        latent_dim: int = 20,
        embedding_size: int = 8,
        noise_sigma: float = 1.,
        sigma_level: int = 10,
        sigma_func: str = 'exp',
        pooling_method: str = 'sum',
        skip_connection: bool = False,
    ):
        super(ScoreMatchingTime, self).__init__(
            tensor_shape=tensor_shape,
            rank=rank,
            h_dim=h_dim,
            act=act,
            dropout=dropout,
            latent_dim=latent_dim,
            embedding_size=embedding_size,
            sigma_func=sigma_func,
            pooling_method=pooling_method,
            skip_connection=skip_connection
        )
        self.noise_sigma = nn.Parameter(torch.tensor(noise_sigma), requires_grad=False)
        self.nosie_sigma_list = torch.from_numpy(np.linspace(noise_sigma, 0.01, sigma_level)).float() #sigma从大到小
        self.setup_energy_()
        self.time_embedding = GaussianFourierProjection(4)

    def setup_energy_(self):
        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # x encoder
        emb_size = 9
        x_enc = MLP(input_dim=emb_size, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # ouput layer
        if self.pooling_method in ['sum', 'attn']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 2 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        output_layer = MLP(input_dim=in_size, output_dim=int(in_size / 2),
                           h_dim=self.h_dim, act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 'output': output_layer})

    def forward(self, idx, t, x, sample=True, return_z=False):
        z = []
        for d in range(self.dim): #self.dim = 2 阶数
            if sample: #diagonal Gaussian and use reparameterization trick
                z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))
            else:
                z_d = self.q_z_mu[d]
            z.append(z_d[idx[:, d]])
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        t = self.time_embedding(t) #size [128]->[128, 8]
        x_exp = torch.cat([x.view(-1, 1), t], -1) #size [128, 1+8]
        x_exp = self.layers['x_enc'](x_exp) #[128, 20]
        z_exp = self.layers['z_enc'](z_ten) #[128, 20]

        if self.pooling_method == 'sum':
            hidden = z_exp + x_exp
        elif self.pooling_method == 'attn':
            hidden = torch.sigmoid(z_exp) * x_exp
        elif self.pooling_method == 'cat':
            hidden = torch.cat([z_exp, x_exp], -1)
        elif self.pooling_method == 'sum_cat':
            hidden = torch.cat([z_exp + x_exp, x_exp], -1)
        else:
            raise RuntimeError('Wrong pooling method!')
        energy = self.layers['output'](hidden).pow(2).sum(-1) #size [128] 保证energy为正

        if return_z:
            return - energy, z 
        else: #-f(x_i, m_i, t_i, \theta)，由于在softplus函数中包含了以个x=exp^(log(x))函数，因此这里不用求exp了
            return - energy 

    def dsm(self, idx, t, x):
        x.requires_grad_(True)
        vector = torch.randn_like(x) * self.noise_sigma
        perturbed_inputs = x + vector
        log_prob = self.forward(idx, t=t, x=perturbed_inputs, sample=True, return_z=False)
        dlogp = self.noise_sigma ** 2 * autograd.grad(log_prob.sum(), perturbed_inputs, create_graph=True)[0]
        kernel = vector
        loss = torch.norm(dlogp + kernel, dim=-1) ** 2
        loss = loss.mean() / 2.
        return loss
    
    def anneal_dsm(self, idx, t, x):
        random_indices = torch.randint(0, len(self.nosie_sigma_list), (x.shape[0],))
        noise_sigmas = self.nosie_sigma_list[random_indices].cuda(x.device)
        x.requires_grad_(True)
        vector = torch.randn_like(x) * noise_sigmas
        perturbed_inputs = x + vector
        log_prob = self.forward(idx, t=t, x=perturbed_inputs, sample=True, return_z=False)
        dlogp = noise_sigmas * autograd.grad(log_prob.sum(), perturbed_inputs, create_graph=True)[0]
        kernel = vector / noise_sigmas
        loss = 0.5 * torch.norm(dlogp + kernel, dim=-1) ** 2
        return loss.mean()
    
    def langevin_sample_posterior(self, idx, time, x0=None, step=100, epsilon=0.01):
        def energy(x):
            log_prob = self.forward(idx, time, x, sample=False, return_z=False)
            return log_prob.sum()

        dev = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype
        if x0 is None:
            x0 = torch.rand(idx.shape[0], device=dev, dtype=dtype)
        x0.requires_grad_(True)
        for _ in range(step):
            score = autograd.grad(energy(x0), x0)[0].detach()
            # x0.data = x0.data + 0.5 * epsilon ** 2 * score + epsilon * torch.randn_like(score)
            x0.data = x0.data + epsilon / 2 * score + math.sqrt(epsilon) * torch.randn_like(score)
        return x0

    def grid_search_posterior(self, idx, time, x_range, epsilon): #x_range=[-0.5, 1.1] epsilon=1e-3
        assert x_range is not None, "Please specify sample range!"
        dev = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype
        x_grid = torch.arange(start=x_range[0], end=x_range[1] + epsilon / 2., step=epsilon,
                              device=dev, dtype=dtype).view(-1, 1).repeat(1, len(idx))  # Grid x Batch
        
        def unnorm_log_prob(x):
            neg_energy = self.forward(idx, time, x, sample=False, return_z=False)
            return neg_energy

        u_log_prob = torch.vmap(unnorm_log_prob, in_dims=0)(x_grid) #[1601, 100]
        idx_est = torch.argmax(u_log_prob, dim=0)
        x_opt = torch.gather(x_grid, 0, idx_est.unsqueeze(0)).squeeze()

        return x_opt

    def predict(self, idx=None, time=None, x_range=None, batch=1e4, 
                sampling=False, x0=None, step=100, epsilon=0.02):
        if idx is not None:
            idx_array = [idx]
            idx_list = None
            idx_full = None
        else:
            idx_array, idx_list, idx_full = self._split_idx(batch)

        tensor_predict = []
        chunk_num = len(idx_array) #1

        for n in range(chunk_num):
            if sampling:  # use Langevin Dynamics sampling
                if x0 is None:
                    x0_n = None
                else:
                    x0_n = x0[idx_list[n]]
                x_hat = self.langevin_sample_posterior(idx=idx_array[n], time=time, x0=x0_n, step=step, epsilon=epsilon)
            else:  # use grid search for MAP
                x_hat = self.grid_search_posterior(idx=idx_array[n], time=time, x_range=x_range, epsilon=epsilon)

            tensor_predict.append(x_hat)

        tensor_predict = torch.cat(tensor_predict).squeeze()
        if idx is not None:
            return tensor_predict

        device = self.q_z_mu[0].device
        dtype = self.q_z_mu[0].dtype
        tensor_full = torch.ones(self.tensor_shape, dtype=dtype, device=device)
        tensor_full[idx_full] = tensor_predict
        return tensor_full
