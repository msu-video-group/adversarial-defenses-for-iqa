import torch
import cv2
import os
import csv
import json
import importlib
import time
import numpy as np
from read_dataset import to_numpy, iter_images, get_batch
from evaluate import compress, predict, write_log, eval_encoded_video, Encoder
from metrics import PSNR, SSIM, MSE, L_inf_dist, MAE
from frozendict import frozendict
from functools import partial
import pandas as pd
from pathlib import Path



def apply_attack(model, attack_callback, dist_images, ref_images=None, metric_range=100, device='cpu', variable_params={}):
    predict_metric = predict if ref_images is None else partial(predict, ref_images)
    clear_metric = predict_metric(dist_images, model=model, device=device)
    if clear_metric is None:
        return None
    
    t0 = time.time()
    if ref_images is None:
        attacked_images = attack_callback(dist_images.clone(), model=model, metric_range=metric_range, device=device, **variable_params)
    else:
        attacked_images = attack_callback(dist_images.clone(), ref_images.clone().detach().contiguous().to(device), model=model, metric_range=metric_range, device=device,
                                          **variable_params)
    attack_time = time.time() - t0
    if attacked_images is None:
        return None
    attacked_metric = predict_metric(attacked_images, model=model, device=device)
    if attacked_metric is None:
        return None
    return attacked_images, clear_metric, attacked_metric, attack_time
    

def run(model, dataset_path, test_dataset, attack_callback, save_path='res.csv', is_fr=False, jpeg_quality=None, codecs=[], batch_size=1, metric_range=100, device='cpu', dump_path=None, dump_freq=500,
        variable_params={}, preset=-1, dataset_save_path=None):
    codecs.append('rawvideo')
    time_sum = 0
    attack_num = 0
    with open(save_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'start_frame', 'end_frame', 'jpeg_quality', 'codec', 'test_dataset', 'clear', 'attacked', 'lower_better', 'rel_gain', 'psnr', 'ssim', 'mse', 'l_inf', 'mae', 'preset']
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
                                    'preset':preset,
                                    'image_name': f'{prev_video_name}',
                                    'start_frame': start,
                                    'end_frame': end,
                                    'clear': None,
                                    'attacked': encoded_metric,
                                    'jpeg_quality': config['jpeg_quality'] if 'jpeg_quality' in config else None,
                                    'codec': config['codec'],
                                    'lower_better' : model.lower_better,
                                    'rel_gain': None,
                                    'test_dataset': test_dataset,
                                    'psnr' : psnr,
                                    'ssim' : ssim,
                                    'mse' : mse,
                                    'l_inf' : linf,
                                    'mae' : mae,
                                    'preset': preset
                                    })
                        os.system(f"vqmt -metr vmaf -metr bfm -orig {prev_path} -in {encoder.fn} -csv-file /artifacts/vqmt_{prev_video_name}_{config['codec']}.csv")
                        os.remove(encoder.fn)
                    encoders = dict()
                    if received_video is not None:
                        for codec in codecs:
                            if is_fr:
                                for q in jpeg_quality:
                                    encoders[frozendict(codec=codec, jpeg_quality=q)] = Encoder(fn=f'{video_name}_{codec}_{q}.mkv', codec=codec)
                            else:
                                encoders[frozendict(codec=codec)] = Encoder(fn=f'{video_name}_{codec}.mkv', codec=codec)
                        prev_path = video_path
                        prev_video_name = video_name
                        is_video = received_video
                local_i = 0
                
            if images is None:
                break
            images = np.stack(images)
            orig_images = images
            images = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)
            images = images.to(device)
            success_attack = True
            if is_fr:    
                for q in jpeg_quality:
                    jpeg_images = compress(orig_images, q, return_torch=True).to(device)
                    attack_result = apply_attack(
                        model,
                        attack_callback,
                        jpeg_images,
                        ref_images=images,
                        metric_range=metric_range,
                        device=device,
                        variable_params=variable_params
                        )
                    if attack_result is not None:
                        attacked_images, clear_metric, attacked_metric, attack_time = attack_result
                    else:
                        success_attack = False
                        break
                    time_sum += attack_time
                    attack_num += 1
                    for config, encoder in encoders.items():
                        if config['jpeg_quality'] == q:
                            encoder.add_frames(to_numpy(attacked_images))
                    writer.writerow({
                        'image_name': f'{video_name}',
                        'start_frame': local_i if is_video else None,
                        'end_frame': (local_i + len(images)) if is_video else None,
                        'clear': clear_metric,
                        'attacked': attacked_metric,
                        'jpeg_quality': q,
                        'codec': None,
                        'lower_better' : model.lower_better,
                        'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                        'test_dataset': test_dataset,
                        'psnr' : PSNR(jpeg_images, attacked_images),
                        'ssim' : SSIM(jpeg_images, attacked_images),
                        'mse' : MSE(jpeg_images, attacked_images),
                        'l_inf' : L_inf_dist(jpeg_images, attacked_images),
                        'mae': MAE(jpeg_images, attacked_images),
                        'preset':preset
                        })
                    
                    if dataset_save_path is not None:
                        cv2.imwrite(os.path.join(dataset_save_path, f'{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(attacked_images).squeeze(0) * 255, cv2.COLOR_RGB2BGR)) 
                    if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                        cv2.imwrite(os.path.join(dump_path, f'{test_dataset}_{fn}-jpeg{q}_orig.png'), cv2.cvtColor(to_numpy(jpeg_images).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(dump_path, f'{test_dataset}_{fn}-jpeg{q}.png'), cv2.cvtColor(to_numpy(attacked_images).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
            else:
                attack_result = apply_attack(
                    model,
                    attack_callback,
                    images.contiguous(),
                    metric_range=metric_range,
                    device=device,
                    variable_params=variable_params
                    )
                if attack_result is not None:
                    attacked_images, clear_metric, attacked_metric, attack_time = attack_result
                else:
                    success_attack = False
                if success_attack:
                    time_sum += attack_time
                    attack_num += 1
                    for config, encoder in encoders.items():
                        encoder.add_frames(to_numpy(attacked_images))
                            
                            
                    writer.writerow({
                        'image_name': f'{video_name}',
                        'start_frame': local_i if is_video else None,
                        'end_frame': (local_i + len(images)) if is_video else None,
                        'clear': clear_metric,
                        'attacked': attacked_metric,
                        'jpeg_quality': None,
                        'codec': None,
                        'lower_better' : model.lower_better,
                        'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                        'test_dataset': test_dataset,
                        'psnr' : PSNR(orig_images, attacked_images),
                        'ssim' : SSIM(orig_images, attacked_images),
                        'mse' : MSE(orig_images, attacked_images),
                        'l_inf' : L_inf_dist(orig_images, attacked_images),
                        'mae' : MAE(orig_images, attacked_images),
                        'preset' : preset
                        })
                    if dataset_save_path is not None:
                        cv2.imwrite(os.path.join(dataset_save_path, f'{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(attacked_images).squeeze(0) * 255, cv2.COLOR_RGB2BGR)) 
                    if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                        cv2.imwrite(os.path.join(dump_path, f'{test_dataset}_{fn}.png'), cv2.cvtColor(to_numpy(attacked_images).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
            local_i += batch_size
            global_i += batch_size
            if not success_attack:
                for config, encoder in encoders.items():
                    encoder.add_frames(to_numpy(orig_images))

    if attack_num == 0:
        return None
    return time_sum / attack_num * 1000

def load_attack_params_json(attack_name, preset_name, presets_path='attack_presets.json'):
    # Used for param picking from json from a predefined set of 10 params
    if preset_name != -1:        
        with open('attack_presets.json') as json_file:
            presets = json.load(json_file)
            cur_attack_config = presets.get(attack_name, None)
            if cur_attack_config is None:
                print(f'[Warning] Attack {attack_name} not found in preset config, using default value')
                return {}
            else:
                param_name = cur_attack_config.get('parameter_name', None)
                default_val = cur_attack_config.get('default_value', None)
                cur_preset = cur_attack_config['presets'].get(str(preset_name), None)
                if param_name is None:
                    print(f'[Warning] Parameter name not specified in json config (attack: {attack_name}), using default values')
                    return {}
                if cur_preset is None:
                    print(f'[Warning] Preset {cur_preset} not found in json config for attack {attack_name}')
                    if default_val is None:
                        print(f'[Warning] Default value not found in json config for attack {attack_name}. Using global default values')
                        return {}
                    else:
                        return {param_name:default_val}
                else:
                    return {param_name:cur_preset}
    else:
        print(f'[Warning] attack: {attack_name} - Preset == -1 was passed: ignoring presets, using global default params')
        return {}

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


def test_main(attack_callback):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--test-dataset", type=str, nargs='+')
    parser.add_argument("--dataset-path", type=str, nargs='+')
    parser.add_argument("--dump-path", type=str, default=None)
    parser.add_argument("--dump-freq", type=int, default=500)
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--save-path", type=str, default='res.csv')
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
    with open('bounds.json') as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(args.metric, None)
        metric_range = 100 if bounds_metric is None else abs(bounds_metric['high'] - bounds_metric['low'])

    # load attack configs
    variable_args = load_attack_params_csv(args.attack, args.metric, args.preset, presets_path=args.presets_csv)
    print(f'Loaded preset: {variable_args}')
    if args.metric in ['paq2piq', 'spaq'] and args.attack == 'onepixel':
        print('Setting batch size in onepixel to 1 as an exception for paq2piq')
        variable_args['batch_size'] = 1
    
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    batch_size = 4 if args.video_metric else 1
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    # Create subfolder to save attacked images
    full_save_path = None
    if args.attacked_dataset_path is not None:
        full_save_path = os.path.join(args.attacked_dataset_path, args.attack)
        full_save_path = os.path.join(full_save_path, args.metric)
        Path(full_save_path).mkdir(parents=True, exist_ok=True)
    
    for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
        mean_time = run(
                        model,
                        dataset_path,
                        test_dataset,
                        attack_callback=attack_callback,
                        save_path=args.save_path,
                        is_fr=is_fr,
                        jpeg_quality=args.jpeg_quality,
                        codecs=args.codecs,
                        batch_size=batch_size,
                        metric_range=metric_range,
                        device=args.device,
                        dump_path=args.dump_path,
                        dump_freq=args.dump_freq,
                        variable_params=variable_args,
                        preset=args.preset,
                        dataset_save_path=full_save_path
                    )
        write_log(args.log_file, test_dataset, mean_time)
