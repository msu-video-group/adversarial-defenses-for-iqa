from __future__ import division, print_function, absolute_import
import numpy as np

import torch
from torchvision import transforms
#from scipy.optimize import OptimizeResult, minimize
#from scipy.optimize.optimize import _status_message
#from scipy._lib._util import check_random_state
#from scipy._lib.six import xrange, string_types
from scipy.optimize import differential_evolution
# Helper functions
#from differential_evolution import differential_evolution
#import helper


_MACHEPS = np.finfo(np.float64).eps



#######################


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])
    
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    #print(imgs.shape)

    for x,img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb
    
    return imgs


def predict_population(xs, img, model, minimize=False, batch_size=32, ref_img=None):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    #print('xs in predict population', xs.shape)
    if xs.ndim == 2:
        #print(f'first set in xs: {xs[:,0]}')
        xs = xs.T
        #print(f'xs new shape: {xs.shape}')
        #print(f'first set in xs after transpose: {xs[0,:]}')
    imgs_perturbed = perturb_image(xs, img)
    predictions = np.zeros(shape=(imgs_perturbed.shape[0]))
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0,l,n):
            yield ndx, min(ndx + n, l), iterable[ndx: min(ndx + n, l)]
    for low, high, x in batch(imgs_perturbed, batch_size):
        cur_batch = torch.from_numpy(x).permute(0,3,1,2).type(torch.FloatTensor).div_(255).to('cuda:0')
        #print(cur_batch.shape)
        with torch.no_grad():
            if ref_img is None:
                predictions[low:high] = model(cur_batch).cpu().detach().numpy().squeeze()
            else:
                cur_ref_batch = torch.tile(ref_img, [cur_batch.shape[0],1,1,1])
                predictions[low:high] = model(cur_ref_batch, cur_batch).cpu().detach().numpy().squeeze()
        torch.cuda.empty_cache()
    predictions = predictions.squeeze()
    #print('preds ', predictions.shape)
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else -predictions


def attack_success(x, img, model, verbose=False, ref_img=None):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, img).squeeze()

    with torch.no_grad():
        if ref_img is None:
            score = model(torch.unsqueeze(transform(attack_image), 0).type(torch.FloatTensor).to('cuda:0')).item()
        else:
            score = model(ref_img.to('cuda:0'),
                          torch.unsqueeze(transform(attack_image), 0).type(torch.FloatTensor).to('cuda:0')).item()
    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if verbose:
        print('Current metric value:', score)
    #if score >= 1.0:
    #    return True
    # NOTE: return None otherwise (not False), due to how Scipy handles its callback function


transform = transforms.Compose([
        transforms.ToTensor()
    ])
def attack_main(img, model, ref_img=None, pixel_count=1, 
           maxiter=75, popsize=400, verbose=False, batch_size=32, minimize=False):
  
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0,img.shape[0]), (0,img.shape[1]), (0,256), (0,256), (0,256)] * pixel_count
    
    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))
    
    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_population(xs, img, model, minimize, batch_size=batch_size, ref_img=ref_img)
    
    def callback_fn(x, convergence):
        return attack_success(x, img,
                              model, verbose, ref_img=ref_img)
    
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False, vectorized=True)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, img)[0]

    res_image = torch.unsqueeze(transform(attack_image), 0).type(torch.FloatTensor)
    res_image = (res_image).data.clamp_(min=0, max=1)
    return res_image


from fgsm_evaluate import test_main 


def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', batch_size=32,
           iters=10, # default 5
           PIXEL_COUNT=3,
           POPSIZE=300
           ):
    verbose=False
    PIXEL_COUNT = int(PIXEL_COUNT)
    POPSIZE = int(POPSIZE)
    
    numpy_image = torch.clamp(compress_image * 255, 0, 255.0).squeeze().permute(1,2,0).cpu().numpy().astype('uint8')
    return attack_main(
        img=numpy_image,
        model=model,
        pixel_count=PIXEL_COUNT,
        maxiter=int(iters),
        popsize=POPSIZE,
        batch_size=batch_size,
        verbose=verbose,
        ref_img=ref_image,
        minimize=model.lower_better
    )
            

if __name__ == "__main__":
   test_main(attack)

