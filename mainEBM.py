import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import torch
import argparse
import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity

from EBM.energy_tensor_cnce import EnergyTDContinuous
from utils.ob_data import get_ob_data
from FNorm.FNorm4Video import read_yuv_video, generate_random_mask_3d
from FNorm.FNorm4RGB import generate_random_mask
from FNorm.FNorm4MSI import load_grayscale_images_from_directory
from EBM.utils import reconstruct_to_tensor
dtype = torch.cuda.FloatTensor

class Trainer:
    """docstring for Trainer."""
    def __init__(self, model, conf, optimizer, print_eval=True, gt_tensor=None, ori_mask=None):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
        self.optimizer = torch.optim.Adam(model.parameters(), lr=conf.train.lr) if optimizer is None else optimizer
        miles = [int(i * conf.train.epoch) for i in conf.train.mile_stones]
        self.miles = miles
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=miles, gamma=0.3)

        self.print_eval = print_eval
        self.eval_interval = conf.train.eval_int
        self.current_epoch = 0
        self.current_iter = 0
        # self.log_test_metric = {'RMSE': [], 'MAE': [], 'MAPE': []}

        self.gt_tensor = gt_tensor.cpu().numpy()
        self.ob_tensor = gt_tensor*(1-ori_mask)
        self.count_tensor = 1-ori_mask

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
        for batch_idx, (x_idx, x_val) in enumerate(bar):
            if torch.cuda.is_available():
                x_idx, x_val = x_idx.cuda(), x_val.type(dtype).cuda()

            if hasattr(self.conf.train, 'data_noise'):
                    x_val += torch.randn_like(x_val) * self.conf.train.data_noise

            vnce = model.loss(x_idx, x_val)
            loss = - vnce
            if batch_idx % 10:
                bar.set_postfix({'Loss': vnce.item()})
                bar.set_description(f'[Iter {batch_idx}]')

            self.optimizer.zero_grad()
            loss.backward()
            if hasattr(self.conf.train, 'grad_clip'):
                if self.current_epoch < self.miles[0] and self.conf.train.grad_clip > 0.:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.conf.train.grad_clip,
                                                   norm_type=self.conf.train.grad_clip_norm)
            self.optimizer.step()
            loss_log.append(vnce.item())
            self.current_iter += 1

        loss_log = np.mean(loss_log)
        self.current_loss = loss_log

    @torch.no_grad()
    def eval_epoch(self, test_loader, phase):
        self.scale = 5.0 # why 5.0?
        model = self.model
        epoch = self.current_epoch
        model.eval()
        
        # x_hat_tot = []
        # x_val_tot = []
        rec_tensor = self.ob_tensor.clone()
        rec_counts = self.count_tensor.clone()
        for _, (x_idx, x_val) in enumerate(test_loader):
            if torch.cuda.is_available():
                x_idx, x_val = x_idx.cuda(), x_val.cuda()
            x_hat = model.predict(x_idx, x_range=[0.0, 1.0], epsilon=1e-2).view(-1)

            batch_tensor, batch_counts = reconstruct_to_tensor(x_hat, x_idx, original_shape=self.model.tensor_shape)
            rec_tensor += batch_tensor
            rec_counts += batch_counts

            # x_hat_tot.append(x_hat * self.scale)
            # x_val_tot.append(x_val * self.scale)

        final_rec = torch.where(rec_counts > 0, rec_tensor / rec_counts.clamp(min=1), torch.zeros_like(rec_tensor))
        final_rec = final_rec.cpu().numpy()
        final_rec = np.clip(final_rec, 0, 1)
        psnr = peak_signal_noise_ratio(self.gt_tensor, final_rec, data_range=1.0)
        ssim = structural_similarity(self.gt_tensor, final_rec, data_range=1.0, channel_axis=2)
        nrmse = normalized_root_mse(self.gt_tensor, final_rec)
        print(f'psnr: {psnr:.4f}, ssim: {ssim:.4f}, nrmse: {nrmse:.4f}')
        
        # x_hat_tot = torch.cat(x_hat_tot)
        # x_val_tot = torch.cat(x_val_tot)
        # rmse = torch.sqrt(torch.mean((x_hat_tot - x_val_tot).pow(2))).item()
        # mae = torch.mean((x_hat_tot - x_val_tot).abs()).item()
        # v = torch.clip(torch.abs(x_val_tot), 0.1, None)
        # diff = torch.abs((x_val_tot - x_hat_tot) / v)
        # mape = 100.0 * torch.mean(diff, axis=-1).mean().item()

        # if self.print_eval:
        #     print(f'Epoch {epoch} - {phase}: RMSE is {rmse:.3f} | MAE is {mae:.3f}.')

        # self.log_test_metric['RMSE'].append(rmse)
        # self.log_test_metric['MAE'].append(mae)
        # self.log_test_metric['MAPE'].append(mape)


def main_run_func(args, conf):
    # read data
    # Video_path = './data/Videos/carphone.yuv'
    # Video_gt = read_yuv_video(Video_path, width=176, height=144, frame_count=100)
    # H, W, C = Video_gt.shape
    # ori_mask = 1.0 - generate_random_mask_3d(H, W, C, visible_ratio=args.ratio) #[H,W,3] True if the value is missing
    # X = torch.from_numpy(Video_gt).type(dtype).cuda()
    
    image_path = './data/misc/4.2.06.tiff' #['4.2.05', '4.2.07', 'house', '4.2.06'] #[Plane Peppers House Sailboat]
    image_gt = np.array(Image.open(image_path)).astype(np.float32) / 255.0        
    H, W, C = image_gt.shape
    ori_mask = 1.0 - generate_random_mask(H, W, visible_ratio=args.ratio)
    X = torch.from_numpy(image_gt).type(dtype).cuda()

    # MSI_path = './data/MSIs/toys'
    # MSI_gt = load_grayscale_images_from_directory(MSI_path)
    # H, W, C = MSI_gt.shape
    # ori_mask = 1.0 - generate_random_mask_3d(H, W, C, visible_ratio=args.ratio)
    # X = torch.from_numpy(MSI_gt).type(dtype).cuda()

    data_loader = get_ob_data(X, ori_mask, batch_size=conf.train.batch_size)
    # model
    model_conf = conf.model
    model = EnergyTDContinuous(
        tensor_shape=[H, W, C],
        rank=model_conf.rank,
        h_dim=model_conf.h_dim,
        act=model_conf.act,
        dropout=model_conf.dropout,
        latent_dim=model_conf.latent_dim,
        embedding_size=model_conf.embedding_size,
        nu=model_conf.nu,
        sigma_func=model_conf.sigma_func,
        noise_sigma=model_conf.noise_sigma,
        pooling_method=model_conf.pooling_method,
        skip_connection=model_conf.skip_connection,
        posdim=model_conf.posdim,
        dtype = dtype
    )
    if torch.cuda.is_available():
        model = model.cuda()

    # trainer
    train_conf = conf.train
    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf.lr, weight_decay=train_conf.weight_decay) #
    trainer = Trainer(model=model, conf=conf, optimizer=optimizer, print_eval=True, gt_tensor=X, ori_mask=ori_mask)
    trainer.train(data_loader['train'], test_loader=data_loader['test'])

    # with open(f'result.txt', 'w') as file:
    #     file.write(json.dumps(trainer.log_test_metric))


def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--ratio', type=float, default=0.1, help='Masking ratio.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # read config
    conf_path = './EBM/conf.yaml'
    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)

    # writer
    main_run_func(args, conf)


if __name__ == "__main__":
    main()
