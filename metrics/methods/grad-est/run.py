#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from fgsm_evaluate import test_main
from tqdm import tqdm

QUALITIES = [100]

def grad_est(metric_model, img, ref_image, sigma, N, n, device='cpu'):
    g = 0
    for i in range(n):
        u = torch.empty((3, N, N)).normal_(mean=0,std=1).to(device)
        if ref_image is not None:
            g += metric_model(get_adv(img.to(device),  sigma*u[None, :]), ref_image).sum()*u
            g -= metric_model(get_adv(img.to(device), -sigma*u[None, :]), ref_image).sum()*u
        else:
            g += metric_model(get_adv(img.to(device),  sigma*u[None, :]).float()).sum()*u
            g -= metric_model(get_adv(img.to(device), -sigma*u[None, :]).float()).sum()*u
    return 1/(2*n*sigma)*g[None, :]        

def get_adv(im, pert):
    return torch.clamp(im + torch.tile(pert, (1, 1, im.shape[2] // pert.shape[2]+ 1, im.shape[
        3] // pert.shape[3]+ 1))[...,:im.shape[2], :im.shape[3]], 0, 1)

def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', eps=0.1):
    #eps = 0.01 # default 0.1
    sigma = 0.001
    N = 32
    n = 20
    eta = 0.1
    max_iters = 250
    sign = -1 if model.lower_better else 1
    lower = torch.clip(compress_image - eps, 0, 1)
    upper = torch.clip(compress_image + eps, 0, 1)
    with torch.no_grad():
        for i_iter in tqdm(range(max_iters)):
            g = grad_est(model, compress_image, ref_image, sigma, N, n, device)
            g = g/torch.linalg.norm(g)
            compress_image = get_adv(compress_image, eta * g * sign)
            compress_image = torch.clip(compress_image, lower, upper)

    return compress_image.squeeze().data.cpu().numpy().transpose(1, 2, 0)

if __name__ == "__main__":
    test_main(attack)
