import torch
import torch.nn as nn
import numpy as np

from .utils import config_activation, gaussian_repar, gaussian_log_prob, MLP
from .energy_tensor_cnce import GaussianFourierProjection
INIT_QZ_SIGMA = 0.2


class ScoreEstimationIdx(nn.Module):
    def __init__(
        self, tensor_shape, rank, h_dim, act, dropout,
        latent_dim, x_emb_size, l_emb_size, sigma_func, 
        sigma_begin, sigma_end, sigma_level,
        pooling_method, skip_connection, dtype):
        super(ScoreEstimationIdx, self).__init__()
        self.dtype = dtype
        self.tensor_shape = list(tensor_shape)
        self.rank = int(rank)
        self.h_dim = h_dim
        self.act = act
        self.dropout = dropout if dropout > 0. else None
        self.latent_dim = latent_dim
        self.x_emb_size = x_emb_size
        if sigma_func == 'exp':
            self.sigma_func = lambda x: torch.exp(x)
        elif sigma_func == 'softplus':
            self.sigma_func = lambda x: nn.functional.softplus(x)
        else:
            raise NotImplementedError
        if sigma_level is not None:
            self.sigmas = torch.from_numpy(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), sigma_level))).float() #sigma从大到小
            # self.sigmas = torch.from_numpy(np.linspace(sigma_begin, sigma_end, sigma_level)).float() #sigma从大到小
            self.label_embed = nn.Embedding(num_embeddings=sigma_level, embedding_dim=l_emb_size)
            self.sigma_level = sigma_level
        else:
            self.sigmas = None
        self.pooling_method = pooling_method
        self.skip_connection = skip_connection
        self.dim = len(tensor_shape)
        if x_emb_size > 1:
            self.x_embedding = GaussianFourierProjection(x_emb_size)
        else:
            self.register_module('x_embedding', None)

        act = config_activation(self.act)
        # z encoder
        z_enc = MLP(input_dim=self.dim * self.rank, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, skip_connection=self.skip_connection)
        # x encoder
        emb_size = 1 if self.x_emb_size == 1 else 2 * self.x_emb_size
        x_enc = MLP(input_dim=emb_size, output_dim=self.latent_dim,
                    h_dim=self.h_dim, act=act, dropout=self.dropout,
                    bn=False, wn=False, sn=False, 
                    skip_connection=self.skip_connection)
        # ouput layer
        if self.pooling_method in ['sum', 'attn', 'none']:
            in_size = self.latent_dim
        elif self.pooling_method in ['cat', 'sum_cat']:
            in_size = 2 * self.latent_dim
        else:
            raise RuntimeError('Wrong pooling method!')
        
        if self.sigmas is not None:
            label_enc = MLP(input_dim=l_emb_size, output_dim=self.latent_dim,
                            h_dim=self.h_dim, act=act, dropout=self.dropout,
                            bn=False, wn=False, sn=False, 
                            skip_connection=self.skip_connection)
            in_size += self.latent_dim
        else:
            label_enc = None
        
        output_layer = MLP(input_dim=in_size, output_dim=1, 
                           h_dim=self.h_dim, act=act, dropout=self.dropout,
                           bn=False, wn=False, sn=False, 
                           skip_connection=self.skip_connection)
        
        self.layers = nn.ModuleDict({'z_enc': z_enc, 'x_enc': x_enc, 'label_enc': label_enc, 'output': output_layer})
        self.setup_q_z_()

    def q_z_sigma(self, d):
        return self.sigma_func(self.q_z_log_sigma[d])
    
    def setup_q_z_(self):
        q_z_mu = []
        q_z_log_sigma = []
        for s in self.tensor_shape:
            q_z_mu.append(nn.Parameter(torch.empty(s, self.rank)))
            q_z_log_sigma.append(nn.Parameter(torch.empty(s, self.rank)))

        self.q_z_mu = nn.ParameterList(q_z_mu) #Size: [dim1, rank] [dim2, rank] 需要优化的参数
        self.q_z_log_sigma = nn.ParameterList(q_z_log_sigma)

        for q in self.q_z_mu:
            torch.nn.init.normal_(q.data, 0., INIT_QZ_SIGMA)
        for q in self.q_z_log_sigma:
            torch.nn.init.normal_(q.data, np.log(np.exp(1.0)-1), INIT_QZ_SIGMA)
        
    def _input_embedding(self, x):
        if self.x_embedding is not None:
            x_exp = self.x_embedding(x.squeeze())
        else:
            x_exp = x.view(-1, 1)
        return x_exp

    def scorenet(self, idx, x, labels=None, sample=True):
        z = []
        for d in range(self.dim):
            if sample:
                # z_d = gaussian_repar(mu=self.q_z_mu[d], sigma=self.q_z_sigma(d))
                z_d = self.q_z_mu[d]
            else:
                z_d = self.q_z_mu[d]
            z.append(z_d[idx[:, d]])
        z_ten = torch.cat(z, -1) #[128, 10] 一个batch 128 个样本，每个样本有self.dim*R = 2 * 5个特征

        # expand input
        x_exp = self._input_embedding(x)
        if labels is not None:
            label_emb = self.label_embed(labels)
            label_exp = self.layers['label_enc'](label_emb)
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
        elif self.pooling_method == 'none':
            hidden = x_exp
        else:
            raise RuntimeError('Wrong pooling method!')
        if labels is not None:
            hidden = torch.cat([hidden, label_exp], -1)
        scores = self.layers['output'](hidden)
        
        return scores

    def anneal_dsm(self, idx, x):
        x = x.unsqueeze(-1)
        n_samples = x.shape[0]
        sigmas_idx = torch.randint(0, self.sigma_level, size=(n_samples, ))
        loss = self.anneal_score_matching(idx, x, sigmas_idx)
        return loss

    def anneal_score_matching(self, idx, samples, labels, anneal_power=2.0):
        used_sigmas = self.sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:]))).type(self.dtype)
        perturbed_samples = samples + torch.randn_like(samples) * used_sigmas #采样数据加噪
        perturbed_labels = labels.cuda()

        target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
        scores = self.scorenet(idx, perturbed_samples, perturbed_labels) #[n, 1]
        loss = 1 / 2.0 * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
        return loss.mean(dim=0)

    @torch.no_grad()
    def predict(self, idx, x_range=[0.0, 1.0], epsilon=None, step_lr=1e-4, n_steps=10):
        n_samples = idx.shape[0]
        pts = torch.rand(n_samples, 1, device=idx.device) * (x_range[1] - x_range[0]) + x_range[0]
        pts = self.anneal_langevin_dynamics(idx, pts, step_lr, n_steps)
        return pts
    
    @torch.no_grad()
    def anneal_langevin_dynamics(self, idx, pts, step_lr, n_steps): #(1e-4, 10)
        # print("before: ", torch.max(pts), torch.min(pts), '\n')
        for c, sigma in enumerate(self.sigmas):
            lables = torch.ones(pts.shape[0], device=pts.device) * c
            lables = lables.long().cuda()
            step_size = step_lr * (sigma / self.sigmas[-1]) ** 2
            for i in range(n_steps):
                # pts = pts + step_size / 2 * self.scorenet(idx, pts, lables, sample=False).detach()
                # pts = pts + torch.randn_like(pts) * np.sqrt(step_size)
                pts = pts + step_size * self.scorenet(idx, pts, lables, sample=False).detach()
                pts = pts + torch.randn_like(pts) * np.sqrt(step_size * 2)
        # print("after: ", torch.max(pts), torch.min(pts), '\n')
        return pts
    # step_lr 和 self.sigma[-1]的数量级需要是正相关的