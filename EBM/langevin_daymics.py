import numpy as np
import torch


def langevin_dynamics(score, pts, lr=0.1, step=1000, save_gif=True):
    pts_list = []
    for i in range(step):
        current_lr = lr
        pts = pts + current_lr / 2 * score(pts).detach()
        pts = pts + torch.randn_like(pts) * np.sqrt(current_lr)
        if save_gif and i % 10 == 0:
            pts_list.append(pts)
    if save_gif:
        return pts_list
    else:
        return pts
    

def anneal_langevin_dynamics(score, pts, sigmas, lr=0.1, n_steps_each=100, save_gif=True, gt=False):
    pts_list = []
    for c, sigma in enumerate(sigmas):
        lables = torch.ones(pts.shape[0], device=pts.device)*c
        lables = lables.long()
        current_lr = lr * (sigma / sigmas[-1]) ** 2
        for i in range(n_steps_each):
            if gt:
                pts = pts + current_lr / 2 * score(pts, sigma).detach()
            else:
                pts = pts + current_lr / 2 * score(pts, lables).detach()
            pts = pts + torch.randn_like(pts) * np.sqrt(current_lr)
            if save_gif and i % 10 == 0:
                pts_list.append(pts)

    if save_gif:
        return pts_list
    else:
        return pts