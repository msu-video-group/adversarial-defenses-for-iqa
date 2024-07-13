#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fgsm_evaluate import test_main

import math
import numpy as np
import torch

import time
import math

import numpy as np
from copy import deepcopy

def get_score(model, x, y):
    sign = -1 if model.lower_better else 1
    if y is None:
        return sign*model(x)
    return sign*model(x, y)

class RSAttack():
    """
    Sparse-RS attacks

    :param predict:           forward pass function
    :param norm:              type of the attack
    :param n_restarts:        number of random restarts
    :param n_queries:         max number of queries (each restart)
    :param eps:               bound on the sparsity of perturbations
    :param seed:              random seed for the starting point
    :param alpha_init:        parameter to control alphai
    :param loss:              loss function optimized ('margin', 'ce' supported)
    :param resc_schedule      adapt schedule of alphai to n_queries
    :param device             specify device to use
    :param log_path           path to save logfile.txt
    :param constant_schedule  use constant alphai
    :param targeted           perform targeted attacks
    :param init_patches       initialization for patches
    :param resample_loc       period in queries of resampling images and
                              locations for universal attacks
    :param data_loader        loader to get new images for resampling
    :param update_loc_period  period in queries of updates of the location
                              for image-specific patches
    """
    
    def __init__(
            self,
            model,
            norm='L0',
            n_queries=5000,
            eps=150,
            p_init=.8,
            n_restarts=1,
            seed=0,
            resc_schedule=True,
            constant_schedule=False,
            init_patches='random_squares'):
        """
        Sparse-RS implementation in PyTorch
        """
        
        self.model = model
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.rescale_schedule = resc_schedule
        self.constant_schedule = constant_schedule
        self.init_patches = init_patches
        self.resample_loc = n_queries // 10
        self.update_loc_period = 4
        self.device = 'cuda'

    def init_hyperparam(self, x):
        assert self.norm in ['L0', 'patches', 'frames',
            'patches_universal', 'frames_universal']

        if self.device is None:
            self.device = x.device
        if self.seed is None:
            self.seed = time.time()
        if 'universal' not in self.norm:
            self.orig_dim = list(x.shape[1:])
            self.ndims = len(self.orig_dim)


    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 'patches' in self.norm:
            if 10 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 8
            elif 500 < it <= 1000:
                p = self.p_init / 16
            elif 1000 < it <= 2000:
                p = self.p_init / 32
            elif 2000 < it <= 4000:
                p = self.p_init / 64
            elif 4000 < it <= 6000:
                p = self.p_init / 128
            elif 6000 < it <= 8000:
                p = self.p_init / 256
            elif 8000 < it:
                p = self.p_init / 512
            else:
                p = self.p_init

        elif 'frames' in self.norm:
            if not 'universal' in self.norm :
                tot_qr = 10000 if self.rescale_schedule else self.n_queries
                p = max((float(tot_qr - it) / tot_qr  - .5) * self.p_init * self.eps ** 2, 0.)
                return 3. * math.ceil(p)
        
            else:
                assert self.rescale_schedule
                its = [200, 600, 1200, 1800, 2500, 10000, 100000]
                resc_factors = [1., .8, .6, .4, .2, .1, 0.]
                c = 0
                while it >= its[c]:
                    c += 1
                return resc_factors[c] * self.p_init
        
        elif 'L0' in self.norm:
            if 0 < it <= 50:
                p = self.p_init / 2
            elif 50 < it <= 200:
                p = self.p_init / 4
            elif 200 < it <= 500:
                p = self.p_init / 5
            elif 500 < it <= 1000:
                p = self.p_init / 6
            elif 1000 < it <= 2000:
                p = self.p_init / 8
            elif 2000 < it <= 4000:
                p = self.p_init / 10
            elif 4000 < it <= 6000:
                p = self.p_init / 12
            elif 6000 < it <= 8000:
                p = self.p_init / 15
            elif 8000 < it:
                p = self.p_init / 20
            else:
                p = self.p_init
        
            if self.constant_schedule:
                p = self.p_init / 2
        
        return p

    def sh_selection(self, it):
        """ schedule to decrease the parameter p """

        t = max((float(self.n_queries - it) / self.n_queries - .0) ** 1., 0) * .75

        return t
    
    def get_init_patch(self, c, s, n_iter=1000):
        if self.init_patches == 'stripes':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device) + self.random_choice(
                [1, c, 1, s]).clamp(0., 1.)
        elif self.init_patches == 'uniform':
            patch_univ = torch.zeros([1, s, s, c]).to(self.device) + self.random_choice(
                [1, c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'random':
            patch_univ = self.random_choice([1, c, s, s]).clamp(0., 1.)
        elif self.init_patches == 'random_squares':
            patch_univ = torch.zeros([1, c, s, s]).to(self.device)
            for _ in range(n_iter):
                size_init = torch.randint(low=1, high=math.ceil(s ** .5), size=[1]).item()
                loc_init = torch.randint(s - size_init + 1, size=[2])
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init] = 0.
                patch_univ[0, :, loc_init[0]:loc_init[0] + size_init, loc_init[1]:loc_init[1] + size_init
                    ] += self.random_choice([c, 1, 1]).clamp(0., 1.)
        elif self.init_patches == 'sh':
            patch_univ = torch.ones([1, c, s, s]).to(self.device)
        
        return patch_univ.clamp(0., 1.)
    
    def attack_single_run(self, x, y):
        with torch.no_grad():
            c, h, w = x.shape[1:]
            
            if self.norm == 'patches':
                ''' assumes square patches '''
                s = int(math.ceil(self.eps ** .5))
                x_best = x.clone()
                x_new = x.clone()
                loc = torch.hstack((torch.randint(h - s, size=[x.shape[0]]), torch.randint(
                    w - s, size=[x.shape[0]]))).reshape((-1, 2))
                patches_coll = torch.zeros([x.shape[0], c, s, s]).to(self.device)
                assert abs(self.update_loc_period) > 1
                loc_t = abs(self.update_loc_period)
                
                # set when to start single channel updates
                it_start_cu = None
                for it in range(0, self.n_queries):
                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    if s_it == 1:
                        break
                it_start_cu = it + (self.n_queries - it) // 2
                    
                # initialize patches
                for counter in range(x.shape[0]):
                    patches_coll[counter] += self.get_init_patch(c, s).squeeze().clamp(0., 1.)
                    x_new[counter, :, loc[counter, 0]:loc[counter, 0] + s,
                        loc[counter, 1]:loc[counter, 1] + s] = patches_coll[counter].clone()
        
                loss_min = get_score(self.model, x_new, y)
                #print(loss_min)
                n_queries = torch.ones(x.shape[0]).to(self.device)
        
                for it in range(1, self.n_queries):
                    # check points still to fool
                    x_curr = self.check_shape(x)
                    patches_curr = self.check_shape(patches_coll)
                    loss_min_curr = loss_min.clone()
                    loc_curr = loc.clone()
        
                    # sample update
                    s_it = int(max(self.p_selection(it) ** .5 * s, 1))
                    p_it = torch.randint(s - s_it + 1, size=[2])
                    sh_it = int(max(self.sh_selection(it) * h, 0))
                    patches_new = patches_curr.clone()
                    x_new = x_curr.clone()
                    loc_new = loc_curr.clone()
                    update_loc = int((it % loc_t == 0) and (sh_it > 0))
                    update_patch = 1. - update_loc
                    if self.update_loc_period < 0 and sh_it > 0:
                        update_loc = 1. - update_loc
                        update_patch = 1. - update_patch
                    for counter in range(x_curr.shape[0]):
                        if update_patch == 1.:
                            # update patch
                            if it < it_start_cu:
                                if s_it > 1:
                                    patches_new[counter, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] += self.random_choice([c, 1, 1])
                                else:
                                    # make sure to sample a different color
                                    old_clr = patches_new[counter, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it].clone()
                                    new_clr = old_clr.clone()
                                    while (new_clr == old_clr).all().item():
                                        new_clr = self.random_choice([c, 1, 1]).clone().clamp(0., 1.)
                                    patches_new[counter, :, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = new_clr.clone()
                            else:
                                assert s_it == 1
                                assert it >= it_start_cu
                                # single channel updates
                                new_ch = self.random_int(low=0, high=3, shape=[1])
                                patches_new[counter, new_ch, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it] = 1. - patches_new[
                                    counter, new_ch, p_it[0]:p_it[0] + s_it, p_it[1]:p_it[1] + s_it]
                                
                            patches_new[counter].clamp_(0., 1.)
                        if update_loc == 1:
                            # update location
                            loc_new[counter] += (torch.randint(low=-sh_it, high=sh_it + 1, size=[2]))
                            loc_new[counter, 0].clamp_(0, h - s)
                            loc_new[counter, 1].clamp_(0, w - s)
                        x_new[counter, :, loc_new[counter, 0]:loc_new[counter, 0] + s,
                            loc_new[counter, 1]:loc_new[counter, 1] + s] = patches_new[counter].clone()
                    
                    # check loss of new candidate
                    loss = get_score(self.model, x_new, y)
                    n_queries += 1
                    #print(loss)
        
                    # update best solution
                    if loss > loss_min_curr:
                        loss_min = loss
                        nimpr = 1
                    else:
                        nimpr = 0
    
                    if nimpr > 0.:
                        patches_coll = patches_new.clone()
                        loc = loc_new.clone()

        
                # apply patches
                for counter in range(x.shape[0]):
                    x_best[counter, :, loc[counter, 0]:loc[counter, 0] + s,
                        loc[counter, 1]:loc[counter, 1] + s] = patches_coll[counter].clone()
        
        return n_queries, x_best

    def perturb(self, x, y):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(x)
        adv = x.clone()
        qr = torch.zeros([len(x)]).to(self.device)

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        for counter in range(self.n_restarts):
            x_to_fool = x.clone()

            qr_curr, adv_curr = self.attack_single_run(x_to_fool, y)
            
            adv = adv_curr.clone()
            qr = qr_curr.clone()


        return qr, adv

    
def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', eps=169):
    max_queries = 10000
    #eps = 49  # default 169
    device='cuda'
    with torch.no_grad():
      model.model.to(device)
      compress_image = compress_image.to(device)
      if ref_image is not None:
          ref_image = ref_image.to(device)
      attack = RSAttack(model, eps=int(eps), norm='patches', n_queries=max_queries)

      res = attack.perturb(compress_image, ref_image)
    return res[1]

if __name__ == "__main__":
    test_main(attack)