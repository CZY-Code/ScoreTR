import json
import torch
import argparse
import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
import os

from EBM.energy4Air import EnergyTDTime
from utils.ob_data import get_air_data


class Trainer:
    """docstring for Trainer."""
    def __init__(
        self,
        model,
        conf,
        optimizer,
        print_eval=True,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=conf.train.lr
        ) if optimizer is None else optimizer
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
            if hasattr(self.conf.train, 'grad_clip'):
                if self.current_epoch < self.miles[0] and self.conf.train.grad_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.conf.train.grad_clip,
                                                   norm_type=self.conf.train.grad_clip_norm)
            self.optimizer.step()
            loss_log.append(loss.item())
            self.current_iter += 1

        loss_log = np.mean(loss_log)
        self.current_loss = loss_log

    @torch.no_grad()
    def eval_epoch(self, test_loader, phase):
        self.scale = 5.0
        model = self.model
        epoch = self.current_epoch

        model.eval()
        x_hat_tot = []
        x_val_tot = []
        for _, (inputs, x_time, x_val) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs, x_val = inputs.cuda(), x_val.cuda()
                x_time = x_time.cuda()

            x_hat = model.predict(inputs, x_time, x_range=[-0.5, 1.1], epsilon=1e-3).view(-1)
            x_hat_tot.append(x_hat * self.scale)
            x_val_tot.append(x_val * self.scale)

        x_hat_tot = torch.cat(x_hat_tot)
        x_val_tot = torch.cat(x_val_tot)

        rmse = torch.sqrt(torch.mean((x_hat_tot - x_val_tot).pow(2))).item()
        mae = torch.mean((x_hat_tot - x_val_tot).abs()).item()

        v = torch.clip(torch.abs(x_val_tot), 0.1, None)
        diff = torch.abs((x_val_tot - x_hat_tot) / v)
        mape = 100.0 * torch.mean(diff, axis=-1).mean().item()

        if self.print_eval:
            print(f'Epoch {epoch} - {phase}: RMSE is {rmse:.3f} | MAE is {mae:.3f}.')

        if phase == 'Test':
            if rmse < self.best_metric['RMSE']:
                self.best_metric['RMSE'] = rmse
                self.best_metric['MAE'] = mae
                self.best_metric['MAPE'] = mape


def main_run_func(args, conf, folds):
    # read data
    file_name = './data'
    data_loader = get_air_data(file_name, batch_size=conf.train.batch_size)
    data_loader = data_loader[folds]

    # model
    model = EnergyTDTime(
        tensor_shape=conf.model.tensor_shape,
        rank=args.rank,
        h_dim=conf.model.h_dim,
        act=conf.model.act,
        dropout=conf.model.dropout,
        latent_dim=conf.model.latent_dim,
        embedding_size=conf.model.embedding_size,
        noise_sigma=args.sigma,
        sigma_level=args.level,
        sigma_func=conf.model.sigma_func,
        pooling_method=conf.model.pooling_method,
        skip_connection=conf.model.skip_connection
    )
    if torch.cuda.is_available():
        model = model.cuda()

    # trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.lr)
    trainer = Trainer(model=model, conf=conf, optimizer=optimizer, print_eval=True)
    trainer.train(data_loader['train'], test_loader=data_loader['test'])

    with open(f'air_result_log.txt', 'a+') as file:
        file.write(f'L: {args.level}, sigma: {args.sigma}, rank: {args.rank}, flod: {folds} '+ json.dumps(trainer.best_metric) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--rank', type=int, default=3, choices=[3, 5, 8, 10])
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--dev', type=int, default=0, help='CUDA ID')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma max')
    parser.add_argument('--level', type=int, default=10, help='num level')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.dev}"
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # read config
    conf_path = './configs/Air_energy_conf.yaml'
    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    # writer
    for i in range(5):
        main_run_func(args, conf, i)
        print(f'Fold {i} finished!')


if __name__ == "__main__":
    main()
