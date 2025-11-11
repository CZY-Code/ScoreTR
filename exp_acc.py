import os
import json
import torch
import argparse
import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from EBM.energy4Acc import ScoreMatchingIdx
from utils.ob_data import get_acc_data
dtype = torch.cuda.FloatTensor

class Trainer:
    """docstring for Trainer."""
    def __init__(self, model, conf, optimizer, print_eval=True):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
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
                if valid_loader is not None:
                    self.eval_epoch(valid_loader, 'Valid')
                if test_loader is not None:
                    self.eval_epoch(test_loader, 'Test')

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
            if hasattr(self.conf.train, 'grad_clip'):
                if self.current_epoch < self.miles[0] and self.conf.train.grad_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.train.grad_clip,
                                                   norm_type=self.conf.train.grad_clip_norm)
            self.optimizer.step()
            loss_log.append(loss.item())
            self.current_iter += 1

        loss_log = np.mean(loss_log)
        self.current_loss = loss_log

    @torch.no_grad()
    def eval_epoch(self, test_loaders, phase):
        val_max = 17.4271
        # val_max = 14.0147
        val_min = 0.6931
        scale = val_max
        self.model.eval()

        rmse_list = []
        mae_list = []
        mape_list = []
        for test_loader in test_loaders:
            x_hat_tot = []
            x_val_tot = []
            for _, (inputs, x_val) in enumerate(test_loader):
                if torch.cuda.is_available():
                    inputs, x_val = inputs.cuda(), x_val.cuda()                

                x_hat = self.model.predict(inputs, x_range=[0.0, 1.01], epsilon=1e-2, step_lr=1e-4, n_steps=100).view(-1)
                # 需要删除torch.no_grad()，否则会导致梯度计算错误
                # x_hat = self.model.predict(inputs, step_lr=1e-2, n_steps=100).view(-1) #noly for energy model

                x_hat_tot.append(x_hat * scale)
                x_val_tot.append(x_val * scale)

            x_hat_tot = torch.cat(x_hat_tot)
            x_val_tot = torch.cat(x_val_tot)

            rmse = torch.sqrt(torch.mean((x_hat_tot - x_val_tot).pow(2)))#.item()
            mae = torch.mean((x_hat_tot - x_val_tot).abs())#.item()

            v = torch.clip(torch.abs(x_val_tot), 0.1, None)
            diff = torch.abs((x_val_tot - x_hat_tot) / v)
            mape = 100.0 * torch.mean(diff, axis=-1).mean()#.item()

            rmse_list.append(rmse)
            mae_list.append(mae)
            mape_list.append(mape)

        rmse = torch.mean(torch.stack(rmse_list)).item()
        mae = torch.mean(torch.stack(mae_list)).item()
        mape = torch.mean(torch.stack(mape_list)).item()

        if self.print_eval:
            print(f'Epoch {self.current_epoch} - {phase}: RMSE is {rmse:.3f} | MAE is {mae:.3f}.')

        if phase == 'Test':
            if rmse <= self.best_metric['RMSE'] and mae <= self.best_metric['MAE']:
                self.best_metric['RMSE'] = rmse
                self.best_metric['MAE'] = mae
                self.best_metric['MAPE'] = mape

def main_run_func(args, conf, folds):
    # read data
    data_loader = get_acc_data(data_path='./data', batch_size=conf.train.batch_size)
    data_loader = data_loader[folds]

    # model
    if args.score:
        # model = ScoreEstimationIdx(
        #     tensor_shape=conf.model.tensor_shape,
        #     rank=args.rank,
        #     h_dim=conf.model.h_dim,
        #     act=conf.model.act,
        #     dropout=conf.model.dropout,
        #     latent_dim=conf.model.latent_dim,
        #     x_emb_size=conf.model.x_emb_size,
        #     l_emb_size=conf.model.l_emb_size,
        #     sigma_func=conf.model.sigma_func,
        #     sigma_begin=conf.model.sigma_begin, 
        #     sigma_end=conf.model.sigma_end, 
        #     sigma_level=conf.model.sigma_level,
        #     pooling_method=conf.model.pooling_method,
        #     skip_connection=conf.model.skip_connection,
        #     dtype = dtype
        # )
        pass
    else:
        model = ScoreMatchingIdx(
            tensor_shape=conf.model.tensor_shape,
            rank=args.rank,
            h_dim=conf.model.h_dim,
            act=conf.model.act,
            dropout=conf.model.dropout,
            latent_dim=conf.model.latent_dim,
            x_emb_size=conf.model.x_emb_size,
            sigma_func=conf.model.sigma_func,
            noise_sigma=args.sigma,
            sigma_level=args.level,
            pooling_method=conf.model.pooling_method,
            skip_connection=conf.model.skip_connection
        )
    print('Model params: ', sum(param.numel() for param in model.parameters())/1e6, 'M')
    if torch.cuda.is_available():
        model = model.cuda()

    # trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.lr, 
                                #  betas = (0.8, 0.999), 
                                 weight_decay = conf.train.weight_decay)
    
    trainer = Trainer(model=model, conf=conf, optimizer=optimizer, print_eval=True)
    trainer.train(data_loader['train'], test_loader=data_loader['test'])

    with open(f'acc_result_log.txt', 'a+') as file:
        file.write(f'L: {args.level}, sigma: {args.sigma}, rank: {args.rank}, flod: {folds} '+ json.dumps(trainer.best_metric) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--rank', type=int, default=3, choices=[3, 5, 8, 10])
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--dev', type=int, default=0, help='CUDA ID')
    parser.add_argument('--score', action='store_true', help='score model')
    parser.add_argument('--sigma', type=float, default=0.1, help='sigma max')
    parser.add_argument('--level', type=int, default=10, help='num level')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.dev}"
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # read config
    if args.score:
        conf_path = './configs/Acc_score_conf.yaml'
    else:
        conf_path = './configs/Acc_energy_conf.yaml'
    
    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    # writer
    for i in range(1):
        main_run_func(args, conf, i)
        print(f'Fold {i} finished!')

if __name__ == "__main__":
    main()
