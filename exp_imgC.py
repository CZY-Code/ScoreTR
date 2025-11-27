import os
import json
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
import argparse
import yaml
from tqdm import tqdm
import cv2
from skimage.metrics import peak_signal_noise_ratio, normalized_root_mse, structural_similarity

from EBM.score4Imgc import ScoreEstimation
from EBM.energy4Imgc import ScoreMatching
from utils.ob_data import get_ob_data
from utils.datautil import read_yuv_video, generate_random_mask_3d, generate_random_mask, load_grayscale_images_from_directory
from EBM.utils import reconstruct_to_tensor
dtype = torch.FloatTensor


class Trainer:
    """docstring for Trainer."""
    def __init__(self, model, conf, gt_tensor, ori_mask, name, visible_ratio, datatype, showChannel):
        super(Trainer, self).__init__()
        self.model = model
        self.conf = conf
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=conf.train.lr, weight_decay=conf.train.weight_decay)
        miles = [int(i * conf.train.epoch) for i in conf.train.mile_stones]
        self.miles = miles
        if self.conf.train.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.miles, gamma=0.3)
        elif self.conf.train.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=conf.train.epoch, eta_min=1e-6)
        else:
            raise RuntimeError('Wrong scheduler!')

        self.eval_interval = conf.train.eval_int
        self.current_epoch = 0
        self.current_iter = 0
        self.max_iter = None

        self.gt_tensor = gt_tensor.cpu().numpy()
        self.ob_tensor = (gt_tensor * (1 - ori_mask)).to(self.model.device)
        self.count_tensor = (1 - ori_mask).to(self.model.device)

        self.best_metric = {'psnr': 0.0, 'ssim': 0.0, 'nrmse': 0.0}
        self.name = name
        self.visible_ratio = visible_ratio
        self.datatype = datatype
        self.showChannel = showChannel

    def train(self, train_loader, valid_loader=None, test_loader=None):
        self.max_iter = len(train_loader) * self.conf.train.epoch
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
                inputs, x_val = inputs.to(self.model.device), x_val.to(self.model.device)
            
            # loss = self.model.dsm(inputs, x_val) #noly for energy model
            loss = self.model.anneal_dsm(inputs, x_val)
            # loss = self.model.anneal_dsm_iter(inputs, x_val, self.current_iter, self.max_iter)
            self.current_iter += 1

            bar.set_postfix({'Loss': loss.item()})
            bar.set_description(f'[Iter {batch_idx}]')

            self.optimizer.zero_grad()
            loss.backward()
            # if hasattr(self.conf.train, 'grad_clip') and self.datatype == 'Video':
            #     if self.current_epoch <= self.miles[0] and self.conf.train.grad_clip > 0.0:
            #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.train.grad_clip,
            #                                         norm_type=self.conf.train.grad_clip_norm)
            self.optimizer.step()
            loss_log.append(loss.item())

        loss_log = np.mean(loss_log)
        self.current_loss = loss_log

    @torch.no_grad()
    def eval_epoch(self, test_loader, phase):
        self.model.eval()

        rec_tensor = self.ob_tensor.clone()
        rec_counts = self.count_tensor.clone()
        for _, (inputs, x_val) in enumerate(test_loader):
            if torch.cuda.is_available():
                inputs, x_val = inputs.to(self.model.device), x_val.to(self.model.device)

            x_hat = self.model.predict(inputs, x_range=[0.0, 1.0], epsilon=1.0/255.0, step_lr=1e-4, n_steps=10).view(-1)

            batch_tensor, batch_counts = reconstruct_to_tensor(x_hat, inputs, original_shape=self.model.tensor_shape)
            rec_tensor += batch_tensor
            rec_counts += batch_counts

        final_rec = torch.where(rec_counts > 0, rec_tensor / rec_counts.clamp(min=1), torch.zeros_like(rec_tensor))
        final_rec = np.clip(final_rec.cpu().detach().numpy(), 0, 1)
        psnr = peak_signal_noise_ratio(self.gt_tensor, final_rec, data_range=1.0)
        ssim = structural_similarity(self.gt_tensor, final_rec, data_range=1.0, channel_axis=2)
        nrmse = normalized_root_mse(self.gt_tensor, final_rec)
        print(f'name: {self.name}, SR: {self.visible_ratio}, psnr: {psnr:.4f}, ssim: {ssim:.4f}, nrmse: {nrmse:.4f}')
        
        if self.showChannel is not None:
            output_folder = os.path.join('output/Ours/inpainting', self.datatype)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, self.name + f'_SR{self.visible_ratio:.2f}_psnr{psnr:.3f}_inpainting.png')
            img = Image.fromarray((final_rec* 255).astype(np.uint8)[..., self.showChannel])
            img.save(output_path)

        with open(f'imgC_{self.datatype}_result.txt', 'a+') as file:
            file.write(f'L: {len(self.model.noise_sigma_list)}, sigma: {self.model.noise_sigma_list[0]:.1f}, name: {self.name}, SR: {self.visible_ratio}, psnr: {psnr:.4f}, ssim: {ssim:.4f}, nrmse: {nrmse:.4f}' + '\n')
        
        if phase == 'Test':
            if psnr > self.best_metric['psnr']:
                self.best_metric['psnr'] = float(psnr)
                self.best_metric['ssim'] = float(ssim)
                self.best_metric['nrmse'] = float(nrmse)


def main():
    parser = argparse.ArgumentParser(description='Tensor completion')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--score', action='store_true', help='score model')
    parser.add_argument('--SR', type=float, default=0.1, help='Masking ratio')
    parser.add_argument('--type', type=str, default='MSI', help='datatype')
    parser.add_argument('--name', type=str, default='toys', help='name')
    parser.add_argument('--dev', type=int, default=2, help='CUDA ID')
    parser.add_argument('--sigma', type=float, default=0.1, help='sigma max')
    parser.add_argument('--level', type=int, default=10, help='num level')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.dev}"
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # read data
    # name, datatype = 'foreman', 'Video' #['foreman', 'carphone']
    # name, datatype = 'Peppers', 'RGB' #['4.2.05', '4.2.07', 'house', '4.2.06'] #[Plane Peppers House Sailboat]
    # name, datatype = 'toys', 'MSI' #['toys', 'flowers']

    if args.type == 'Video':
        showChannel = 0
        file_path = os.path.join('data/Videos', args.name+'.yuv')
        Video_gt = read_yuv_video(file_path, width=176, height=144, frame_count=100)
        H, W, C = Video_gt.shape
        ori_mask = 1.0 - generate_random_mask_3d(H, W, C, visible_ratio=args.SR) #[H,W,3] True if the value is missing
        X = torch.from_numpy(Video_gt).type(dtype)
    elif args.type == 'RGB':
        showChannel = [0, 1, 2]
        file_path = os.path.join('data/misc', args.name+'.tiff') 
        image_gt = np.array(Image.open(file_path)).astype(np.float32) / 255.0
        H, W, C = image_gt.shape
        ori_mask = 1.0 - generate_random_mask(H, W, visible_ratio=args.SR)
        X = torch.from_numpy(image_gt).type(dtype)
    elif args.type == 'MSI':
        showChannel = [15, 25, 30]
        file_path = os.path.join('data/MSIs', args.name)
        MSI_ori_gt = load_grayscale_images_from_directory(file_path)
        MSI_gt = np.zeros((256, 256, 31), dtype=MSI_ori_gt.dtype)
        for i in range(MSI_ori_gt.shape[-1]):
            MSI_gt[:, :, i] = cv2.resize(MSI_ori_gt[:, :, i], (256, 256), interpolation=cv2.INTER_LINEAR)
        H, W, C = MSI_gt.shape
        ori_mask = 1.0 - generate_random_mask_3d(H, W, C, visible_ratio=args.SR)
        X = torch.from_numpy(MSI_gt).type(dtype)
    else:
        raise NotImplementedError
    
    # read config
    if args.score:
        conf_path = f'./configs/ImgC_{args.type}_score_conf.yaml'
    else:
        conf_path = f'./configs/ImgC_{args.type}_energy_conf.yaml'

    with open(conf_path) as f:
        conf = yaml.full_load(f)
    conf = OmegaConf.create(conf)
    
    data_loader = get_ob_data(X, ori_mask, batch_size=conf.train.batch_size)
    if args.score:
        model = ScoreEstimation(
            tensor_shape = [H, W, C],
            rank = conf.model.rank,
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
            skip_connection = conf.model.skip_connection,
            posdim = conf.model.posdim,
            dtype = dtype,
            device = device,
            rff_alpha = conf.model.rff_alpha,
            rff_sigma = conf.model.rff_sigma
        )
    else:
        model = ScoreMatching(
            tensor_shape = [H, W, C],
            rank = conf.model.rank,
            h_dim = conf.model.h_dim,
            act = conf.model.act,
            dropout = conf.model.dropout,
            latent_dim = conf.model.latent_dim,
            x_emb_size = conf.model.x_emb_size,
            sigma_func = conf.model.sigma_func,
            # sigma_begin =conf.model.sigma_begin,
            sigma_begin = args.sigma, 
            sigma_end = conf.model.sigma_end, 
            # sigma_level =conf.model.sigma_level,
            sigma_level = args.level,
            pooling_method = conf.model.pooling_method,
            skip_connection = conf.model.skip_connection,
            posdim = conf.model.posdim,
            dtype = dtype,
            device = device,
            rff_alpha = conf.model.rff_alpha,
            rff_sigma = conf.model.rff_sigma
        )
    print('Model params: ', sum(param.numel() for param in model.parameters())/1e6, 'M')
    
    if torch.cuda.is_available():
        model = model.to(device)
    
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"模型总参数量: {total_params/1e6:.2f} M")
    # print(f"可训练参数量: {trainable_params/1e6:.2f} M")
    # trainer
    trainer = Trainer(model=model, conf=conf, gt_tensor=X, ori_mask=ori_mask,
                      name=args.name, visible_ratio=args.SR, datatype=args.type, showChannel=showChannel)
    trainer.train(data_loader['train'], test_loader=data_loader['test'])

    with open(f'imgC_result_log.txt', 'a+') as file:
        file.write(f'L: {args.level}, sigma: {args.sigma}, name: {args.name}, rank: {conf.model.rank}'+ json.dumps(trainer.best_metric) + '\n')

if __name__ == "__main__":
    main()
