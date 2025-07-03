import os
import torch
import argparse
import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

from EBM.energy4SimC import EnergyTDTime
from utils.ob_data import get_continuous_data
dtype = torch.cuda.FloatTensor

class Trainer:
    """docstring for Trainer."""
    def __init__(self, model, conf, optimizer, test_dt, test_inter, print_eval=True):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
        self.test_dt = test_dt
        self.test_inter = test_inter
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

    def train(self, train_loader, valid_loader=None, test_loader=None):
        bar = tqdm(range(self.conf.train.epoch), desc='[Epoch 0]')
        for epoch in bar:
            bar.set_description(f'[Epoch {epoch}]')
            self.train_epoch(train_loader)
            self.scheduler.step()
            bar.set_postfix({'Loss': self.current_loss})

            is_eval = epoch % self.eval_interval == 0 or epoch == self.conf.train.epoch - 1

            if is_eval:
                self.eval_epoch('Test')

            self.current_epoch += 1

    def train_epoch(self, data_loader):
        model = self.model
        model.train()

        loss_log = []
        bar = tqdm(data_loader, desc='[Iter 0]', leave=False)
        for batch_idx, (inputs, x_time, x_val) in enumerate(bar):
            if torch.cuda.is_available():
                inputs, x_val = inputs.cuda(), x_val.cuda()
                x_time = x_time.cuda()

            # loss = model.dsm(inputs, x_time, x_val)
            loss = model.anneal_dsm(inputs, x_time, x_val)
            
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
    def eval_epoch(self, phase):
        self.model.eval()
        N = 200
        val_max, val_min = 2.0, -0.6
        scale = 10.0
        x_time = torch.linspace(0, 1, 200).cuda()  # [200,]
        # idx_list = [[1, 2], [2, 1], [1, 1], [2, 2]]
        idx_list = [[1, 2], [2, 3], [3, 4], [4, 5]]
        color_list = [['#bccbe8', '#1d73b6'], ['#c6dfb8', '#24a645'], ['#fcd5b4', '#f27830'], ['#d7cae4','#8768a6']]
        # 创建图表和子图（建议显式设置背景色）
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_facecolor('#f0f0f0')  #背景
        for (begin, end) in self.test_inter:
            ax.axvspan(xmin=begin, xmax=end, facecolor='#cdc9c1', alpha=0.5)

        for i, idx in enumerate(idx_list):
            inputs = torch.tensor(idx, dtype=torch.long).view(1, 2).expand(N, -1).cuda()
            x_val = torch.stack([self.test_dt[i*64 + idx[0]*8 + idx[1]][2] for i in range(200)])
            # x_idx = torch.stack([self.test_dt[i*64 + idx[0]*8 + idx[1]][0] for i in range(200)])
            x_hat = self.model.predict(inputs, x_time, x_range=[val_min, val_max], epsilon=1e-3).view(-1)

            pred_val = x_hat.cpu().numpy() * scale
            true_val = x_val.cpu().numpy() * scale
            time = x_time.cpu().numpy()

            ax.plot(time, true_val, color=color_list[i][0], alpha=1.0, linewidth=2, label=rf'${{x}}_{{{idx[0]},{idx[1]}}}$')
            ax.plot(time, pred_val, color=color_list[i][1], alpha=0.8, linewidth=2, label=rf'$\tilde{{x}}_{{{idx[0]},{idx[1]}}}$')

            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2)) # x轴主网格间隔
            ax.yaxis.set_major_locator(ticker.MultipleLocator(2.0))
            ax.set_xlabel('Time', fontsize=10)
            # ax.set_ylabel('Density', fontsize=10)

        handles, labels = ax.get_legend_handles_labels()
        train_patch = Rectangle((0, 0), 1, 1, facecolor='#f0f0f0', label='Train')
        test_patch = Rectangle((0, 0), 1, 1, facecolor='#cdc9c1', alpha=0.5, label='Test')
        handles.extend([train_patch, test_patch])
        labels.extend(['Train', 'Test'])

        # ax.legend(handles=handles, labels=labels, fontsize=9, loc='upper left')  # 建议指定图例位置
        ax.grid(True, linestyle='--', alpha=0.8, color='#ffffff')  # 降低网格透明度
        fig.tight_layout() 
        
        # fig_legend = plt.figure(figsize=(3, 1))
        # ax_legend = fig_legend.add_subplot(111)
        # ax_legend.axis('off')  # 隐藏坐标轴
        # ax_legend.legend(handles, labels, fontsize=9, loc='center', ncol=10)
        # fig_legend.savefig('legend.png', dpi=300, bbox_inches='tight')
        
        plt.show()


        
def main_run_func(args, conf):
    data_loader, test_dt, test_list = get_continuous_data(batch_size=conf.train.batch_size, shape = conf.model.tensor_shape)

    # model
    if args.score:
        pass
    else:
        model = EnergyTDTime(
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
        skip_connection = conf.model.skip_connection
        )
    if torch.cuda.is_available():
        model = model.cuda()

    # trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.lr, weight_decay = conf.train.weight_decay)
    trainer = Trainer(model=model, conf=conf, optimizer=optimizer, print_eval=True, test_dt=test_dt, test_inter=test_list)
    trainer.train(data_loader)


def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--rank', type=int, default=5, choices=[3, 5, 8, 10])
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--score', action='store_true', help='score model')
    parser.add_argument('--dev', type=int, default=0, help='CUDA ID')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.dev}"

    # read config
    if args.score:
        conf_path = './configs/SimC_score_conf.yaml'   
    else:
        conf_path = './configs/SimC_energy_conf.yaml'
    
    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    main_run_func(args, conf)

if __name__ == "__main__":
    main()
