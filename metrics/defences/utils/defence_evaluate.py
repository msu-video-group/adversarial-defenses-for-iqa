import torch
import cv2
import os
import csv
import json
import importlib
#import time
import numpy as np
from read_dataset import to_numpy, iter_images, get_batch
from evaluate import compress, predict, write_log, eval_encoded_video, Encoder
from metrics import PSNR, SSIM, MSE, L_inf_dist, MAE
from frozendict import frozendict
from functools import partial
import pandas as pd
from pathlib import Path
from defence_dataset import KoniqAttackedDataset
from defence_scoring_methods import calc_scores_defence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import torchvision
import re
np.random.seed(int(time()))
# fr cols
FR_COLS = ['ssim', 'mse', 'psnr', 'l_inf', 'mae']
# Columns in resulting dataframe with raw values
RAW_RESULTS_COLS = ['image_name', 'clear', 'attacked', 'defended-clear', 'defended-attacked','mos'] + \
    [f'{x}_clear_defended-clear' for x in FR_COLS] + \
    [f'{x}_clear_attacked' for x in FR_COLS] + \
    [f'{x}_clear_defended-attacked' for x in FR_COLS] + \
    [f'{x}_attacked_defended-attacked' for x in FR_COLS] + \
    [f'{x}_defended-clear_defended-attacked' for x in FR_COLS]

def calc_frs(src, dest):
    fr_vals = {
                'mse':[],
                'mae':[],
                'ssim':[],
                'psnr':[],
                'l_inf':[]
            }
    for i in range(len(src)):
        fr_vals['mse'].append(MSE(src[i].unsqueeze(0), dest[i].unsqueeze(0)))
        fr_vals['mae'].append(MAE(src[i].unsqueeze(0), dest[i].unsqueeze(0)))
        fr_vals['ssim'].append(SSIM(src[i].unsqueeze(0), dest[i].unsqueeze(0)))
        fr_vals['psnr'].append(PSNR(src[i].unsqueeze(0), dest[i].unsqueeze(0)))
        fr_vals['l_inf'].append(L_inf_dist(src[i].unsqueeze(0), dest[i].unsqueeze(0)))
    return fr_vals

def run(defence, model, dataloader, is_fr=False, device='cpu', dump_path=None, dump_freq=500, save_freq=-1,
        defence_preset=-1, dataset_save_path=None, atk_name=''):
    total_time = 0
    cur_result_df = pd.DataFrame(columns=RAW_RESULTS_COLS)

    if is_fr:
        raise NotImplementedError('Defence Evaluation for FR target metrics is not implemented yet.')
    
    batch_idx = 0
    for clear_imgs, attacked_imgs, img_names, moses in tqdm(dataloader):
        cur_bs = len(clear_imgs)
        clear_imgs = clear_imgs.to(device)
        attacked_imgs = attacked_imgs.to(device)
        with torch.no_grad():
            # apply defence
            time_st = time()
            # random seed equal for both clear and attacked
            torch_seed = np.random.randint(low=0, high=999999)
            torch.manual_seed(torch_seed)
            defended_attacked = defence(attacked_imgs.clone())
            torch.manual_seed(torch_seed)
            defended_clear = defence(clear_imgs.clone())
            total_time += time() - time_st
            # target metric scores
            clear_scores = model(clear_imgs)
            attacked_scores = model(attacked_imgs)
            defended_attacked_scores = model(defended_attacked)
            defended_clear_scores = model(defended_clear)

            # Resize to original shape, if defended image is compared with undefended
            h, w = clear_imgs.shape[2], clear_imgs.shape[3]
            resize_op = torchvision.transforms.Resize((h,w))
            # FRs between different pairs of images
            # undef. vs undef
            fr_clear_attacked = calc_frs(clear_imgs, attacked_imgs)
            # def. vs def.
            fr_defended_clear_defended_attacked = calc_frs(defended_clear, defended_attacked)
            # undef. vs def.
            fr_clear_defended_clear = calc_frs(clear_imgs, resize_op(defended_clear))
            fr_clear_defended_attacked = calc_frs(clear_imgs, resize_op(defended_attacked))
            fr_attacked_defended_attacked = calc_frs(attacked_imgs, resize_op(defended_attacked))
        # print('Clear input shape: ', clear_imgs.shape)
        # print('Clear shape: ', clear_scores.shape)
        # print('Clear tensor: ', clear_scores)
        for i in range(cur_bs):
            row = {
                'image_name':Path(img_names[i]).stem + '.jpg',
                'mos':float(moses[i]),
                }
            if cur_bs == 1:
                row['clear'] = float(clear_scores.item())
                row['attacked'] = float(attacked_scores.item())
                row['defended-attacked'] = float(defended_attacked_scores.item())
                row['defended-clear'] = float(defended_clear_scores.item())
            else:
                row['clear'] = float(clear_scores[i])
                row['attacked'] = float(attacked_scores[i])
                row['defended-attacked'] = float(defended_attacked_scores[i])
                row['defended-clear'] = float(defended_clear_scores[i])
            
            for fr_metric in FR_COLS:
                row[f'{fr_metric}_clear_attacked'] = fr_clear_attacked[fr_metric][i]
                row[f'{fr_metric}_clear_defended-clear'] = fr_clear_defended_clear[fr_metric][i]
                row[f'{fr_metric}_clear_defended-attacked'] = fr_clear_defended_attacked[fr_metric][i]
                row[f'{fr_metric}_attacked_defended-attacked'] = fr_attacked_defended_attacked[fr_metric][i]
                row[f'{fr_metric}_defended-clear_defended-attacked'] = fr_defended_clear_defended_attacked[fr_metric][i]
            cur_result_df.loc[len(cur_result_df)] = row
        
        # Dumping debug images
        if dump_path is not None and batch_idx % dump_freq == 0:
            #print(attacked_imgs[0].unsqueeze(0).shape)
            fn = Path(img_names[0]).stem
            cv2.imwrite(os.path.join(dump_path, f'attacked_{atk_name}_{fn}.png'), cv2.cvtColor(to_numpy(torch.clamp(attacked_imgs[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dump_path, f'clear_{atk_name}_{fn}.png'), cv2.cvtColor(to_numpy(torch.clamp(clear_imgs[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dump_path, f'defended-clear_{atk_name}_{fn}.png'), cv2.cvtColor(to_numpy(torch.clamp(defended_clear[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dump_path, f'defended-attacked_{atk_name}_{fn}.png'), cv2.cvtColor(to_numpy(torch.clamp(defended_attacked[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
        # Saving all defended-attacked, if required
        if dataset_save_path is not None:
            if save_freq == -1:
                for i in range(cur_bs):
                    fn = Path(img_names[i]).stem
                    cv2.imwrite(os.path.join(dataset_save_path, f'{fn}.png'),
                                cv2.cvtColor(to_numpy(torch.clamp(defended_attacked[i],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
            else:
                if batch_idx % save_freq == 0:
                    fn = Path(img_names[0]).stem
                    cv2.imwrite(os.path.join(dataset_save_path, f'{fn}.png'),
                                cv2.cvtColor(to_numpy(torch.clamp(defended_attacked[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
        batch_idx += 1
    return cur_result_df, total_time


    


def load_defence_params_json(defence_name, preset_name, presets_path='./defence_presets.json', use_default=False, raise_error=True):
    # Used for param picking from json from a predefined set of 10 params
    if not use_default:        
        with open(presets_path) as json_file:
            presets = json.load(json_file)
            cur_attack_config = presets.get(defence_name, None)
            if cur_attack_config is None:
                if raise_error:
                    raise ValueError('Defence name not found in JSON.')
                print(f'[Warning] Attack {defence_name} not found in preset config, using default init value')
                return {}
            else:
                param_name = cur_attack_config.get('parameter_name', None)
                default_val = cur_attack_config.get('default_value', None)
                cur_preset = cur_attack_config['presets'].get(str(preset_name), None)
                if param_name is None:
                    if raise_error:
                        raise ValueError(f'Parameter name not found in JSON.')
                    print(f'[Warning] Parameter name not specified in json config (attack: {defence_name}), using default init values')
                    return {}
                if cur_preset is None:
                    if raise_error:
                        raise ValueError(f'{cur_preset} not found in JSON.')
                    print(f'[Warning] Preset {cur_preset} not found in json config for attack {defence_name}')
                    if default_val is None:
                        print(f'[Warning] Default value not found in json config for attack {defence_name}. Using global default values')
                        return {}
                    else:
                        return {param_name:default_val}
                else:
                    return {param_name:cur_preset}
    else:
        print(f'[Warning] attack: {defence_name} - use-default-preset==True was passed: ignoring presets, using global default params')
        return {}


def test_main(defence_class):
    import argparse
    parser = argparse.ArgumentParser()
    # Essential
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--defence", type=str, required=True, help='Name of the tested defence.')
    parser.add_argument("--batch-size", type=int, default=1, help='Batch size to test defences with.')
    parser.add_argument("--save-defended", type=int, default=0, help='Whether to save defended images or not. 1 -- yes, 0 -- no.')
    parser.add_argument("--save-freq", type=int, default=-1)
    # Now defence is always tested on all available presets
    # parser.add_argument("--attacks-preset", type=int, default=0, help='Preset of the attacks on which defence will be tested.')
    # parser.add_argument("--run-all-attack-presets", type=int, default=1, help='If set to 1, attacks-preset argument is ignored, \
    #                     and defence is tested on all available attacks presets.')

    # Paths
    parser.add_argument("--attacked-dataset-path", type=str, default=None, help='Path to directory where attacked dataset is stored. \
                        Should have following structure: *this_path*/*attack_name*/*metric_name*/*images*.png')
    parser.add_argument("--src-dir", type=str, required=True, help='Path to a dataset with source images. \
                        Should be a directory with images.')
    parser.add_argument("--mos-path", type=str, required=True, help='Path to a CSV with MOSes for all source images.')
    parser.add_argument("--save-path", type=str, default='res.csv', help='Save path for CSV with raw results on all images.') 
    parser.add_argument("--defended-dataset-path", type=str, default=None, help='Path to directory where defended dataset will be stored. \
                        Will have following structure: *this_path*/*attacks_preset*/*attack_name*/*metric_name*/*defended_images*.png') 
    parser.add_argument("--dump-path", type=str, default=None, help='Directory where debug images will be dumped.')
    parser.add_argument("--log-file", type=str, default=None, help='Path to CSV file with logs: defence evaluation scores and time.')

    # Defence preset args
    parser.add_argument("--defence-preset", type=int, default=0)
    parser.add_argument("--use-default-preset", type=int, default=1, help='If set to 1, dafault initialization value is used. \
                        Otherwise, defence-preset from presets-json-path')
    parser.add_argument("--presets-json-path", type=str, default='./defence_presets.json',
                        help='Path to JSON file with presets info for defences.')

    # Misc
    parser.add_argument("--dump-freq", type=int, default=500)
    parser.add_argument('--video-metric', action='store_true')
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*', help='Currently not used.')
    parser.add_argument("--codecs", type=str, nargs='*', default=[], help='Currently not used.')
    # parser.add_argument("--test-dataset", type=str, nargs='+')
    # parser.add_argument("--dataset-path", type=str, nargs='+')

    args = parser.parse_args()
    print(f'===Defence: {args.defence}===\n CI/CD launched test job successfully')
    print(f'Defence Class: {defence_class}')
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    with open('bounds.json') as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(args.metric, None)
        metric_range = 100 if bounds_metric is None else abs(bounds_metric['high'] - bounds_metric['low'])

    # # load defence configs and init it
    variable_args = load_defence_params_json(args.defence, args.defence_preset,
                                            presets_path=args.presets_json_path, use_default=args.use_default_preset == 1)
    print(f'Loaded preset: {variable_args}')
    defence_obj = defence_class(**variable_args)
    
    # load target metric
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    batch_size = args.batch_size
    dump_freq = args.dump_freq
    save_freq = args.save_freq
    # FPR and SPAQ don't work with bs > 1
    if args.metric in ['fpr', 'spaq']:
        batch_size = 1
        dump_freq *= args.batch_size
        save_freq *= args.batch_size
    # PAQ2PIQ exception.
    # TODO: fix later
    if args.metric == 'paq2piq':
        model.model.as_loss = False
        print('Applying as_loss=False to paq2piq model')
    print(f'Batch size: {batch_size}')
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    # Find all available attack presets:
    atk_presets = [Path(x).name for x in Path(args.attacked_dataset_path).iterdir() if 
                   Path(x).is_dir() and 'preset' in str(x)]
    atk_preset_nums = [re.findall(r'\d+|-\d+', str(x))[0] for x in atk_presets]
    print('Found preset folders: ', atk_presets)
    print('Preset numbers: ', atk_preset_nums)
    # DEBUG, REMOVE LATER
    # attacks = [attacks[0]]
    #

    # Create subfolder to save defended images
    # full_save_path = None
    # if args.save_defended == 1 and args.defended_dataset_path is not None:
    #     full_save_path = os.path.join(args.defended_dataset_path, f'preset_{args.attacks_preset}')
    #     Path(full_save_path).mkdir(parents=True, exist_ok=True)
    
    raw_results_df = pd.DataFrame(columns=RAW_RESULTS_COLS)
    scores_df = pd.DataFrame(columns=['attack','attack_preset', 'defence_preset', 'score', 'value'])
    total_time = 0
    total_calls = 0
    for atk_preset_name, atk_preset_num in zip(atk_presets, atk_preset_nums):
        print(f'======Current preset: {atk_preset_num} ======')
        # Find available attacks in attacked dataset directory
        preset_path = os.path.join(args.attacked_dataset_path, atk_preset_name)
        attacks = [Path(x).stem for x in Path(preset_path).iterdir() if 
                   Path(x).is_dir() and Path(os.path.join(str(x), args.metric)).exists()]
        
        print(f'Attacks found for attack preset {atk_preset_name} and metric {args.metric}: {attacks}')
        for atk in attacks:
            # Directory where defended-attacked images will be saved, if required
            if args.save_defended == 1 and args.defended_dataset_path is not None:
                cur_defended_dataset_save_path = os.path.join(args.defended_dataset_path, atk_preset_name)
                cur_defended_dataset_save_path = os.path.join(cur_defended_dataset_save_path, atk)
                cur_defended_dataset_save_path = os.path.join(cur_defended_dataset_save_path, args.metric)
                Path(cur_defended_dataset_save_path).mkdir(parents=True, exist_ok=True)
            else:
                cur_defended_dataset_save_path = None
            print('Saving Path for defended images: ', cur_defended_dataset_save_path)
            #cur_attacked_dir = os.path.join(args.attacked_dataset_path, f'preset_{args.attacks_preset}')
            cur_attacked_dir = os.path.join(preset_path, atk)
            cur_attacked_dir = os.path.join(cur_attacked_dir, args.metric)
            cur_dataset = KoniqAttackedDataset(src_dir=args.src_dir, dest_dir=cur_attacked_dir, mos_path=args.mos_path, return_mos=True)
            cur_dloader = DataLoader(cur_dataset, batch_size=batch_size, shuffle=False)
            cur_raw_results, cur_time = run(
                                defence=defence_obj, 
                                model=model, 
                                dataloader=cur_dloader,
                                is_fr=is_fr, device=args.device, dump_freq=dump_freq, save_freq=save_freq, dump_path=args.dump_path,
                                dataset_save_path=cur_defended_dataset_save_path,
                                atk_name=atk)
            total_time += cur_time
            total_calls += 2 * len(cur_dataset)

            cur_raw_results['attack'] = atk
            cur_raw_results['attack_preset'] = atk_preset_num
            cur_raw_results['attack_preset'] = cur_raw_results['attack_preset'].astype(int)
            cur_raw_results['defence_preset'] = args.defence_preset
            cur_raw_results['defence_preset'] = cur_raw_results['defence_preset'].astype(int)
            # Merge raw results
            raw_results_df = pd.concat([raw_results_df, cur_raw_results]).reset_index(drop=True)

            # Calculate scores on raw results (on each attack separately)
            cur_scores = calc_scores_defence(cur_raw_results, metric_range=metric_range)
            cur_scores.loc[len(cur_scores)] = {'score':'mean_time', 'value':cur_time / (2 * len(cur_dataset))}
            cur_scores['attack'] = atk
            cur_scores['attack_preset'] = atk_preset_num
            cur_scores['attack_preset'] = cur_scores['attack_preset'].astype(int)
            cur_scores['defence_preset'] = args.defence_preset
            cur_scores['defence_preset'] = cur_scores['defence_preset'].astype(int)
            scores_df = pd.concat([scores_df, cur_scores]).reset_index(drop=True)
        
            # TODO: time measurement (done)
    # Scores on all attacks
    total_scores = calc_scores_defence(raw_results_df, metric_range=metric_range)
    total_scores.loc[len(total_scores)] = {'score':'mean_time', 'value':total_time / total_calls}
    total_scores['attack'] = 'total'
    total_scores['attack_preset'] = 'total'
    #total_scores['attack_preset'] = total_scores['attack_preset'].astype(int)
    total_scores['defence_preset'] = args.defence_preset
    total_scores['defence_preset'] = total_scores['defence_preset'].astype(int)
    scores_df = pd.concat([total_scores, scores_df]).reset_index(drop=True)
    # Save CSVs
    scores_df.reset_index(drop=True).to_csv(args.log_file)
    raw_results_df.reset_index(drop=True).to_csv(args.save_path)
