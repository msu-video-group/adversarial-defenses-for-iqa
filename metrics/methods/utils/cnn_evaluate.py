import cv2
import os
import csv
import json
import importlib
import time
import numpy as np
from read_dataset import iter_images, get_batch, to_numpy, to_torch
from evaluate import compress, predict, write_log, eval_encoded_video, Encoder
from metrics import PSNR, SSIM, MSE, L_inf_dist, MAE
from torch import nn
import torch
from pathlib import Path
import pandas as pd
from frozendict import frozendict

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_type='batch', act_type='selu'):
        super(UnetGenerator, self).__init__()
        self.name = 'unet'
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(ngf)
            self.norm2 = nn.BatchNorm2d(ngf * 2)
            self.norm4 = nn.BatchNorm2d(ngf * 4)
            self.norm8 = nn.BatchNorm2d(ngf * 8)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(ngf)
            self.norm2 = nn.InstanceNorm2d(ngf * 2)
            self.norm4 = nn.InstanceNorm2d(ngf * 4)
            self.norm8 = nn.InstanceNorm2d(ngf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        if act_type == 'selu':
            self.act = nn.SELU(True)
        else:
            self.act = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 512 x 1024
        e1 = self.conv1(input)
        # state size is (ngf) x 256 x 512
        e2 = self.norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 128 x 256
        e3 = self.norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 64 x 128
        e4 = self.norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 32 x 64
        e5 = self.norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 16 x 32
        e6 = self.norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 8 x 16
        e7 = self.norm8(self.conv7(self.leaky_relu(e6)))
        # state size is (ngf x 8) x 4 x 8
        # No batch norm on output of Encoder
        e8 = self.conv8(self.leaky_relu(e7))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 2 x 4
        d1_ = self.dropout(self.norm8(self.dconv1(self.act(e8))))
        # state size is (ngf x 8) x 4 x 8
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dropout(self.norm8(self.dconv2(self.act(d1))))
        # state size is (ngf x 8) x 8 x 16
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dropout(self.norm8(self.dconv3(self.act(d2))))
        # state size is (ngf x 8) x 16 x 32
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.norm8(self.dconv4(self.act(d3)))
        # state size is (ngf x 8) x 32 x 64
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.norm4(self.dconv5(self.act(d4)))
        # state size is (ngf x 4) x 64 x 128
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.norm2(self.dconv6(self.act(d5)))
        # state size is (ngf x 2) x 128 x 256
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.norm(self.dconv7(self.act(d6)))
        # state size is (ngf) x 256 x 512
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.act(d7))
        # state size is (nc) x 512 x 1024
        output = self.tanh(d8)
        return output
    
    
def normalize_and_scale(delta_im, mode='train'):
    delta_im = (delta_im) * 10.0/255.0
    return delta_im


def run(model, cnn_weights, dataset_path, train_dataset, test_dataset, amplitude=[0.2], is_fr=False, jpeg_quality=None, codecs=[], batch_size=1, save_path='res.csv', device='cpu', dump_path=None, dump_freq=500,
        preset=-1, dataset_save_path=None):
    codecs.append('rawvideo')
    
    netG = UnetGenerator(3, 3, 64, norm_type='instance', act_type='relu').to(device)
    if device == 'cpu':
        netG.load_state_dict(torch.load(cnn_weights, map_location='cpu'))
    else:
        netG.load_state_dict(torch.load(cnn_weights))
    netG.eval()
    
    time_sum = 0
    attack_num = 0
    
    
    with open(save_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'start_frame', 'end_frame', 'train_dataset', 'test_dataset', 'clear', 'attacked',
                      'jpeg_quality', 'codec', 'lower_better', 'amplitude', 'rel_gain', 'psnr', 'ssim', 'mse', 'l_inf', 'mae', 'preset']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        video_iter = iter_images(dataset_path)
        prev_path = None
        prev_video_name = None
        encoders = dict()
        is_video = None
        global_i = 0
        while True:
            images, video_name, fn, video_path, video_iter, received_video = get_batch(video_iter, batch_size)
            
            if is_video is None:
                is_video = received_video
            
            if video_name != prev_video_name:
                if is_video:
                    for config, encoder in encoders.items():
                        encoder.close()
                        if config['codec'] != 'rawvideo':
                            for encoded_metric, start, end, psnr, ssim, mse, linf, mae in eval_encoded_video(model, encoder.fn, orig_video_path=prev_path, is_fr=is_fr, batch_size=batch_size, device=device):
                                writer.writerow({
                                    'image_name': f'{prev_video_name}',
                                    'start_frame': start,
                                    'end_frame': end,
                                    'clear': None,
                                    'attacked': encoded_metric,
                                    'jpeg_quality': config['jpeg_quality'] if 'jpeg_quality' in config else None,
                                    'codec': config['codec'],
                                    'lower_better' : model.lower_better,
                                    'rel_gain': None,
                                    'train_dataset': train_dataset,
                                    'test_dataset': test_dataset,
                                    'amplitude': config['amplitude'],
                                    'psnr' : psnr,
                                    'ssim' : ssim,
                                    'mse' : mse,
                                    'l_inf' : linf,
                                    'mae' : mae,
                                    'preset' : preset
                                    })
                        os.system(f"vqmt -metr vmaf -metr bfm -orig {prev_path} -in {encoder.fn} -csv-file /artifacts/vqmt_{prev_video_name}_{config['codec']}.csv")  
                        os.remove(encoder.fn)
                    encoders = dict()
                    if received_video is not None:
                        for codec in codecs:
                            if is_fr:
                                for q in jpeg_quality:
                                    for k in amplitude:
                                        encoders[frozendict(codec=codec, jpeg_quality=q, amplitude=k)] = Encoder(fn=f'{video_name}_{codec}_{q}_{k}.mkv', codec=codec)
                            else:
                                for k in amplitude:
                                    encoders[frozendict(codec=codec, amplitude=k)] = Encoder(fn=f'{video_name}_{codec}_{k}.mkv', codec=codec)
                        prev_path = video_path
                        prev_video_name = video_name
                        is_video = received_video
                local_i = 0
                
            if images is None:
                break
            images = np.stack(images)
            orig_images = images
            
            h, w = orig_images[0].shape[:2]
            h, w = h // 256, w // 256
            h, w = h * 256, w * 256
            success_attack = True
            
            if is_fr:    
                for q in jpeg_quality:
                    jpeg_images = compress(orig_images, q)
                    clear_metric = predict(orig_images, jpeg_images, model=model, device=device)
                    if clear_metric is None:
                        success_attack = False
                        break
                    for k in amplitude:
                        timage = to_torch(jpeg_images[:,:h,:w,:], device=device)
                        if (timage.shape[2] < 256) and (timage.shape[3] < 256):
                            cur_h = timage.shape[2]
                            cur_w = timage.shape[3]
                            new_timage = torch.zeros([1, 3, 256, 256])
                            new_timage[:,:,:cur_h,:cur_w] = a
                            timage = new_timage
                            timage = timage.to(device)
                        elif (timage.shape[2] < 256):
                            cur_w = timage.shape[3]
                            new_timage = torch.zeros([1, 3, 256, cur_w])
                            new_timage[:,:,:cur_h,:cur_w] = a
                            timage = new_timage
                            timage = timage.to(device)
                        elif (timage.shape[3] < 256):
                            cur_h = timage.shape[2]
                            new_timage = torch.zeros([1, 3, cur_h, 256])
                            new_timage[:,:,:cur_h,:cur_w] = a
                            timage = new_timage
                            timage = timage.to(device)
                        
                        
                        t0 = time.time()
                        delta_im = netG(timage)
                        delta_im = normalize_and_scale(delta_im) 
                        delta_im = delta_im.data.cpu().numpy().transpose(0, 2, 3, 1) 
                        delta = np.tile(delta_im,(1, orig_images.shape[1]//h + 1, orig_images.shape[2]//w + 1, 1))[:, :orig_images.shape[1], :orig_images.shape[2], :]
                        attacked_images = jpeg_images + delta * k * 10
                        attacked_images = np.clip(attacked_images, 0, 1)
                        time_sum += time.time() - t0
                        attack_num += 1
                        
                        attacked_metric = predict(orig_images, attacked_images, model=model, device=device)
                        
                        if attacked_metric is None:
                            success_attack = False
                            break
                        for config, encoder in encoders.items():
                            if config['jpeg_quality'] == q and config['amplitude'] == k:
                                encoder.add_frames(to_numpy(attacked_images))
                            
                        writer.writerow({
                            'image_name': f'{video_name}',
                            'start_frame': local_i if is_video else None,
                            'end_frame': (local_i + len(images)) if is_video else None,
                            'clear': clear_metric,
                            'attacked': attacked_metric,
                            'jpeg_quality': q,
                            'codec': None,
                            'lower_better': model.lower_better,
                            'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                            'train_dataset': train_dataset,
                            'test_dataset': test_dataset,
                            'amplitude': k,
                            'psnr' : PSNR(jpeg_images, attacked_images),
                            'ssim' : SSIM(jpeg_images, attacked_images),
                            'mse' : MSE(jpeg_images, attacked_images),
                            'l_inf' : L_inf_dist(jpeg_images, attacked_images),
                            'mae' : MAE(jpeg_images, attacked_images),
                            'preset' : preset
                            })
                        
                        if dataset_save_path is not None:
                            cv2.imwrite(os.path.join(dataset_save_path, f'{Path(fn).stem}.png'),
                                        cv2.cvtColor((to_numpy(attacked_images).squeeze(0) * 255).astype(np.float32), cv2.COLOR_RGB2BGR)) 
                        
                        if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                            cv2.imwrite(os.path.join(dump_path, f'{train_dataset}_{test_dataset}_{k}_{fn}-jpeg{q}.png'), attacked_images * 255)
                    if not success_attack:
                        break        
                    if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                        cv2.imwrite(os.path.join(dump_path, f'{train_dataset}_{test_dataset}_{fn}-jpeg{q}_orig.png'), jpeg_images * 255)
            else:
                clear_metric = predict(orig_images, model=model, device=device)
                if clear_metric is None:
                    success_attack = False
                else:
                    for k in amplitude:
                        timage = to_torch(orig_images[:,:h,:w,:], device=device)
                        if (timage.shape[2] < 256) or (timage.shape[3] < 256):
                            cur_h = timage.shape[2]
                            cur_w = timage.shape[3]
                            new_timage = torch.zeros([1, 3, 256, 256])
                            new_timage[:,:,:cur_h,:cur_w] = a
                            timage = new_timage
                            timage = timage.to(device)
                        elif (timage.shape[2] < 256):
                            cur_w = timage.shape[3]
                            new_timage = torch.zeros([1, 3, 256, cur_w])
                            new_timage[:,:,:cur_h,:cur_w] = a
                            timage = new_timage
                            timage = timage.to(device)
                        elif (timage.shape[3] < 256):
                            cur_h = timage.shape[2]
                            new_timage = torch.zeros([1, 3, cur_h, 256])
                            new_timage[:,:,:cur_h,:cur_w] = a
                            timage = new_timage
                            timage = timage.to(device)
                        
                        
                        t0 = time.time()
                        delta_im = netG(timage)
                        delta_im = normalize_and_scale(delta_im) 
                        delta_im = delta_im.data.cpu().numpy().transpose(0, 2, 3, 1) 
                        delta = np.tile(delta_im,(1, orig_images.shape[1]//h + 1, orig_images.shape[2]//w + 1, 1))[:, :orig_images.shape[1], :orig_images.shape[2], :]
                        attacked_images = orig_images + delta * k * 10
                        attacked_images = np.clip(attacked_images, 0, 1)
                        time_sum += time.time() - t0
                        attack_num += 1
    
                        attacked_metric = predict(attacked_images, model=model, device=device)
                        if attacked_metric is None:
                            success_attack = False
                            break
                        for config, encoder in encoders.items():
                            if config['amplitude'] == k:
                                encoder.add_frames(to_numpy(attacked_images))
                        writer.writerow({
                            'image_name': f'{video_name}',
                            'start_frame': local_i if is_video else None,
                            'end_frame': (local_i + len(images)) if is_video else None,
                            'clear': clear_metric,
                            'attacked': attacked_metric,
                            'jpeg_quality': None,
                            'codec': None,
                            'lower_better': model.lower_better,
                            'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                            'train_dataset': train_dataset,
                            'test_dataset': test_dataset,
                            'amplitude': k,
                            'psnr' : PSNR(orig_images, attacked_images),
                            'ssim' : SSIM(orig_images, attacked_images),
                            'mse' : MSE(orig_images, attacked_images),
                            'l_inf' : L_inf_dist(orig_images, attacked_images),
                            'mae' : MAE(orig_images, attacked_images),
                            'preset' : preset
                            })
                        if dataset_save_path is not None:
                            cv2.imwrite(os.path.join(dataset_save_path, f'{Path(fn).stem}.png'),
                                        cv2.cvtColor((to_numpy(attacked_images).squeeze(0) * 255).astype(np.float32), cv2.COLOR_RGB2BGR)) 
                        if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                            cv2.imwrite(os.path.join(dump_path, f'{train_dataset}_{test_dataset}_{k}_{fn}.png'), attacked_images * 255)
            local_i += batch_size
            global_i += batch_size
            if not success_attack:
                for config, encoder in encoders.items():
                    encoder.add_frames(to_numpy(orig_images))
    if attack_num == 0:
        return 0
    return time_sum / attack_num * 1000


def train_main(train_callback):
    import importlib
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-train", type=str, nargs='+')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, nargs='+')
    parser.add_argument("--save-dir", type=str, default="./")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    with open('bounds.json') as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(args.metric, None)
        metric_range = 100 if bounds_metric is None else abs(bounds_metric['high'] - bounds_metric['low'])
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    for train_dataset, path_train in zip(args.train_dataset, args.path_train):
        unet_state_dict = train_callback(model, path_train, batch_size=args.batch_size, is_fr=is_fr, jpeg_quality=args.jpeg_quality, metric_range=metric_range, device=args.device)
        torch.save(np.array([0]), os.path.join(args.save_dir, f'{train_dataset}.png'))
        torch.save(unet_state_dict, os.path.join(args.save_dir, f'{train_dataset}.npy'))

def load_attack_params_csv(attack_name, metric_name, preset_name, presets_path='picked_presets.csv'):
    '''
    Used to load params for attack from CSV file with already picked parameters and their values for all bounds
    '''
    assert preset_name >= 0
      
    presets = pd.read_csv(presets_path)
    cur_attack_preset = presets[presets.attack == attack_name]
    cur_attack_preset = cur_attack_preset[cur_attack_preset.metric == metric_name]
    cur_attack_preset = cur_attack_preset[cur_attack_preset.category == preset_name]
    if len(cur_attack_preset) == 0:
        raise ValueError(f'Attack: {attack_name}; Metric: {metric_name}; preset: {preset_name} -- NOT FOUND IN CSV {presets_path}')
    
    if len(cur_attack_preset) > 1:
        print(f'[Warning] More than one entry is found for: Attack: {attack_name}; Metric: {metric_name}; preset: {preset_name}. \
               Using first entry.')
    param_val = cur_attack_preset['param_val'].values[0]
    param_name = cur_attack_preset['param_name'].values[0]
    print('Loaded entry:')
    print(cur_attack_preset)

    return {param_name:param_val}  

def test_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--uap-path", type=str, nargs='+')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, nargs='+')
    parser.add_argument("--test-dataset", type=str, nargs='+')
    parser.add_argument("--dataset-path", type=str, nargs='+')
    parser.add_argument("--save-path", type=str, default='res.csv')
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--amplitude", type=float, default=[0.2], nargs='+')
    parser.add_argument("--dump-path", type=str, default=None)
    parser.add_argument("--dump-freq", type=int, default=500)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument('--video-metric', action='store_true')
    parser.add_argument("--codecs", type=str, nargs='*', default=[])
    parser.add_argument("--preset", type=int, default=-1)
    parser.add_argument("--attack", type=str, default='undefined')
    parser.add_argument("--attacked-dataset-path", type=str, default=None, help='Path to directory where attacked dataset will be stored. \
                        Images will be saved to *this_path*/*attack_name*/*metric_name*/.')
    parser.add_argument("--presets-csv", type=str, default='./picked_presets_v1.csv', help='Path to CSV file with preset info.')
    args = parser.parse_args()
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    
    # load attack configs
    variable_args = load_attack_params_csv(args.attack, args.metric, args.preset, presets_path=args.presets_csv)
    print(f'Loaded preset: {variable_args}')

    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    batch_size = 4 if args.video_metric else 1

    # Create subfolder to save attacked images
    full_save_path = None
    if args.attacked_dataset_path is not None:
        full_save_path = os.path.join(args.attacked_dataset_path, args.attack)
        full_save_path = os.path.join(full_save_path, args.metric)
        Path(full_save_path).mkdir(parents=True, exist_ok=True)
    
    for train_dataset, uap_path in zip(args.train_dataset, args.uap_path):
        for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
            mean_time = run(
                model,
                uap_path,
                dataset_path,
                train_dataset,
                test_dataset,
                amplitude=[variable_args['Amplitude']],
                is_fr=is_fr,
                jpeg_quality=args.jpeg_quality,
                codecs=args.codecs,
                batch_size=batch_size,
                save_path=args.save_path,
                device=args.device,
                dump_path=args.dump_path,
                dump_freq=args.dump_freq,
                preset=args.preset,
                dataset_save_path=full_save_path
            )
            write_log(args.log_file, test_dataset, mean_time)
