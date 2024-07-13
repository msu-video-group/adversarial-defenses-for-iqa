#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader


from tqdm import tqdm

from read_dataset import MyCustomDataset
from evaluate import jpeg_generator 


from uap_evaluate import train_main


def train(metric_model, path_train, batch_size=8, is_fr=False, jpeg_quality=None, metric_range=100, device='cpu'):
    ds_train = MyCustomDataset(path_gt=path_train, device=device)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    eps = 0.1
    lr = 0.001
    n_epoch = 5
    universal_noise = torch.zeros((1, 3, 256, 256)).to(device)
    universal_noise += 0.0001
    universal_noise = Variable(universal_noise, requires_grad=True)
    optimizer = torch.optim.Adam([universal_noise], lr=lr)
    sign = -1 if metric_model.lower_better else 1
    for epoch in range(n_epoch):
        for u in tqdm(range(len(dl_train))):
            y = next(iter(dl_train))
            for orig_image, jpeg_image in jpeg_generator(y, jpeg_quality):
                if is_fr:
                    res = (jpeg_image.to(device) + universal_noise).type(torch.FloatTensor).to(device)
                    res.data.clamp_(min=0.,max=1.)
                    y = y.to(device)
                    score = metric_model(y, res)
                else:
                    res = (y + universal_noise).to(device)
                    res.data.clamp_(min=0.,max=1.)
                    score = metric_model(res)
                if score is None:
                    break
                loss = 1 - score.mean() * sign / metric_range
                
                optimizer.zero_grad()
                loss.backward()
                universal_noise.grad = torch.nan_to_num(universal_noise.grad)
                optimizer.step()
                universal_noise.data.clamp_(min=-eps, max=eps)
                #universal_noise = Variable(universal_noise, requires_grad=True)
    return universal_noise.squeeze().data.cpu().numpy().transpose(1, 2, 0)



if __name__ == "__main__":
    train_main(train)
