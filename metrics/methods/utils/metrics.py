from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import numpy as np
from read_dataset import to_numpy


def PSNR(x, y):
    return peak_signal_noise_ratio(to_numpy(x), to_numpy(y), data_range=1)


def SSIM(x, y):
    ssim_sum = 0
    for img1, img2 in zip(to_numpy(x), to_numpy(y)):
        ssim = structural_similarity(img1, img2, channel_axis=2, data_range=1)
        ssim_sum += ssim
    return ssim_sum / len(x)

def MSE(x, y):
    return mean_squared_error(to_numpy(x), to_numpy(y))

def L_inf_dist(x, y):
    return np.max(np.abs(to_numpy(x).astype(np.float32) - to_numpy(y).astype(np.float32)))

def MAE(x, y):
    return np.abs(to_numpy(x).astype(np.float32) - to_numpy(y).astype(np.float32)).mean()