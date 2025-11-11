import os
import torch
import argparse
import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import beta, gaussian_kde, norm, expon
from scipy.integrate import fixed_quad

from EBM.energy4Sim import ScoreMatchingIdx
from utils.ob_data import get_sim_data
dtype = torch.cuda.FloatTensor

class Trainer:
    """docstring for Trainer."""
    def __init__(self, model, conf, optimizer, dist, Z_list, print_eval=True):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
        self.dist = dist
        self.Z_list = Z_list
        self.optimizer = torch.optim.Adam(model.parameters(), lr = conf.train.lr) if optimizer is None else optimizer
        miles = [int(i * conf.train.epoch) for i in conf.train.mile_stones]
        self.miles = miles

        if self.conf.train.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.miles, gamma=0.3)
        elif self.conf.train.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=conf.train.epoch, eta_min=1e-6)
        else:
            raise RuntimeError('Wrong scheduler!')

        self.print_eval = print_eval
        self.eval_interval = conf.train.eval_int
        self.current_epoch = 0
        self.current_iter = 0
        self.best_metric = {'RMSE': 1e2, 'MAE': 1e2, 'MAPE': 1e2}

    def train(self, train_loader, dataset):
        bar = tqdm(range(self.conf.train.epoch), desc='[Epoch 0]')
        for epoch in bar:
            bar.set_description(f'[Epoch {epoch}]')
            self.train_epoch(train_loader)
            self.scheduler.step()
            bar.set_postfix({'Loss': self.current_loss})

            is_eval = epoch % self.eval_interval == 0 or epoch == self.conf.train.epoch - 1
            if is_eval:
                self.eval_epoch(dataset, 'Test')

            self.current_epoch += 1

    def train_epoch(self, data_loader):
        self.model.train()
        loss_log = []
        bar = tqdm(data_loader, desc='[Iter 0]', leave=False)
        for batch_idx, (inputs, x_val) in enumerate(bar):
            if torch.cuda.is_available():
                inputs, x_val = inputs.cuda(), x_val.cuda()

            # loss = self.model.dsm(inputs, x_val) #noly for energy model
            loss = self.model.anneal_dsm(inputs, x_val)
            
            if batch_idx % 10:
                bar.set_postfix({'Loss': loss.item()})
                bar.set_description(f'[Iter {batch_idx}]')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_log.append(loss.item())
            self.current_iter += 1

        loss_log = np.mean(loss_log)
        self.current_loss = loss_log

    @torch.no_grad()
    def eval_epoch(self, dataset, phase = 'Test'):
        if self.dist == 'beta':
            val_max, val_min = 1.0, 0.0 #BUG
        elif self.dist == 'MoG':
            val_max, val_min = 1.9, -0.9
        elif self.dist == 'exp':   
            val_max, val_min = 13.3, -1.0
        self.model.eval()
        epsilon = 1e-3
        idx = [4, 6]
        _idx = torch.tensor(idx, dtype=torch.long).view(1, 2)
    
        def integrand(x):
            x = torch.from_numpy(x).type(dtype).cuda()
            _idx_ = _idx.expand(x.shape[0], 2).cuda()
            neg_energy = self.model.forward(_idx_, x, sample=False, return_z=False)
            return torch.exp(neg_energy).cpu().numpy()
        Z, _ = fixed_quad(integrand, val_min, val_max, n=100)

        x_val_tot = torch.stack([dataset[i*64 + idx[0]*8 + idx[1]][1] for i in range(200)]) #[200]
        x_val_tot = x_val_tot.cpu().detach().numpy()
        u_log_prob, x_hat_tot = self.model.grid_search_posterior_idx(_idx, x_range=[val_min, val_max], epsilon=epsilon)
        pred_prob = torch.exp(u_log_prob) / Z
        # print('sum of pred prob: ', torch.sum(pred_prob * epsilon).item())
        pred_prob = pred_prob.cpu().detach().numpy()
        x_hat_tot = x_hat_tot.cpu().detach().numpy()
        
        if True:
            val_kde = gaussian_kde(x_val_tot)
            x_range = np.linspace(val_min, val_max, 200)
            val_density = val_kde(x_range)

            # 计算理论Beta分布(α=1.25, β=1.2)的PDF
            if self.dist == 'beta':
                alpha = (self.Z_list[0] @ self.Z_list[1]).cpu().detach().numpy()[tuple(idx)]
                true_density = beta.pdf(x_range, a=alpha, b=5.0)
            elif self.dist == 'MoG':
                mu = (self.Z_list[0] @ self.Z_list[1]).cpu().detach().numpy()[tuple(idx)]
                mu_1 = np.cos(mu)
                mu_2 = np.sin(mu)
                true_density = 0.6 * norm.pdf(x_range, loc=mu_1, scale=0.1) + \
                               0.4 * norm.pdf(x_range, loc=mu_2, scale=0.25)
            elif self.dist == 'exp':
                lambda_param = (self.Z_list[0] @ self.Z_list[1]).cpu().detach().numpy()[tuple(idx)]
                true_density = expon.pdf(x_range, scale=1/lambda_param)
            
            # 创建图表和子图（建议显式设置背景色）
            fig, ax = plt.subplots(figsize=(4, 3))
            fontsize = 16
            # 设置图表整体背景色（外围区域）
            # fig.patch.set_facecolor('#ffffff')  # 白色 
            # 设置绘图区域（坐标轴内）的背景色（可选，增强层次感）
            ax.set_facecolor('#f0f0f0')  # 浅灰背景

            ax.plot(x_range, true_density, color='#24a645', linewidth=2, label=r'Truth')
            ax.fill_between(x_range, true_density, alpha=0.2, color='#c6dfb8')

            ax.plot(x_range, val_density, color='#f27830', linewidth=2, label=r'Obs')
            ax.fill_between(x_range, val_density, alpha=0.2, color='#fcd5b4')

            ax.plot(x_hat_tot, pred_prob, color='#1d73b6', linewidth=2, label='ScoreTR')
            ax.fill_between(x_hat_tot, pred_prob, alpha=0.2, color='#bccbe8')

            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0)) # x轴主网格间隔
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.tick_params(axis='x', labelsize=fontsize)  # x轴刻度数值字体大小
            ax.tick_params(axis='y', labelsize=fontsize)  # y轴刻度数值字体大小
            ax.set_xlabel('x', fontsize=fontsize)
            ax.set_ylabel('Density', fontsize=fontsize)

            # 其他设置
            ax.legend(fontsize=fontsize, loc='upper right')  # 建议指定图例位置
            ax.grid(True, linestyle='--', alpha=0.8, color='#ffffff')  # 降低网格透明度
            fig.tight_layout()
            plt.savefig(f'{self.dist}.png', dpi=300, bbox_inches='tight')
            plt.show()

def main_run_func(args, conf):
    data_loader, dataset, Z_list = get_sim_data(dist=args.dist, batch_size=conf.train.batch_size, 
                                                   R=args.rank, shape = conf.model.tensor_shape)

    # model
    if args.score:
        pass
    else:
        model = ScoreMatchingIdx(
        tensor_shape = conf.model.tensor_shape,
        rank = args.rank,
        h_dim = conf.model.h_dim,
        act = conf.model.act,
        dropout = conf.model.dropout,
        latent_dim = conf.model.latent_dim,
        x_emb_size = conf.model.x_emb_size,
        sigma_func = conf.model.sigma_func,
        sigma_begin = conf.model.sigma_begin, 
        sigma_end = conf.model.sigma_end, 
        sigma_level = conf.model.sigma_level,
        pooling_method = conf.model.pooling_method,
        skip_connection = conf.model.skip_connection)
    if torch.cuda.is_available():
        model = model.cuda()

    # trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.lr, weight_decay = conf.train.weight_decay)
    trainer = Trainer(model=model, conf=conf, optimizer=optimizer, print_eval=True, dist=args.dist, Z_list=Z_list)
    trainer.train(data_loader, dataset)


def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--rank', type=int, default=5, choices=[3, 5, 8, 10])
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--score', action='store_true', help='score model')
    parser.add_argument('--dev', type=int, default=0, help='CUDA ID')
    parser.add_argument('--dist', type=str, default='beta', choices=['beta', 'MoG', 'exp'], help='distribution type')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.dev}"

    # read config
    if args.score:
        conf_path = './configs/Sim_score_conf.yaml'   
    else:
        conf_path = './configs/Sim_energy_conf.yaml'
    
    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    main_run_func(args, conf)

if __name__ == "__main__":
    main()
