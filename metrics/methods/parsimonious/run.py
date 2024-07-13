#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fgsm_evaluate import test_main

import itertools
import math
import numpy as np
import heapq
import torchvision.transforms as T
import torch

def get_loss(model, compress_image, ref_image):
  sign = -1 if model.lower_better else 1
  if ref_image is None:
    return sign*model(compress_image)
  else:
    return sign*model(compress_image, ref_image)

class LocalSearchHelper(object):
  """A helper for local search algorithm.
  Note that since heapq library only supports min heap, we flip the sign of loss function.
  """

  def __init__(self, model, epsilon, max_iters):
    """Initalize local search helper.
    
    Args:
      model: model
      loss_func: str, the type of loss function
      epsilon: float, the maximum perturbation of pixel value
    """
    # Hyperparameter setting 
    self.epsilon = epsilon
    self.max_iters = max_iters
    self.model = model

  def _flip_noise(self, noise, block):
    """Flip the sign of perturbation on a block.
    Args:
      noise: numpy array of size [3, 256, 256, 3], a noise
      block: [upper_left, lower_right, channel], a block
    
    Returns:
      noise: numpy array of size [3, 256, 256], an updated noise 
    """
    noise_new = noise.clone()
    upper_left, lower_right, channel = block 
    noise_new[channel, upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]] *= -1
    return noise_new

  def perturb(self, image, ref_image, noise, blocks):
    """Update a noise with local search algorithm.
    
    Args:
      image: numpy array of size [3, 299, 299], an original image
      noise: numpy array of size [3, 256, 256], a noise
      blocks: list, a set of blocks

    Returns: 
      noise: numpy array of size [3, 256, 256], an updated noise
      num_queries: int, the number of queries
      curr_loss: float, the value of loss function
    """
    # Class variables
    self.width = image.shape[2]
    self.height = image.shape[3]
    # Local variables
    priority_queue = []
    num_queries = 0
    
    # Check if a block is in the working set or not
    A = np.zeros([len(blocks)], np.int32)
    for i, block in enumerate(blocks):
      upper_left, _, channel = block
      x = upper_left[0]
      y = upper_left[1]
      # If the sign of perturbation on the block is positive,
      # which means the block is in the working set, then set A to 1
      if noise[channel, x, y] > 0:
        A[i] = 1

    # Calculate the current loss
    image_batch = perturb_image(image, noise)
    
    loss = get_loss(self.model, image_batch, ref_image)
    #print(loss)
    num_queries += 1
    curr_loss = loss
  
    # Main loop
    for _ in range(self.max_iters):
      # Lazy greedy insert
      indices,  = np.where(A==0)
      
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = torch.zeros([bend-bstart, 3, self.width, self.height]).to(image.device)
        noise_batch = torch.zeros([bend-bstart, 3, 256, 256]).to(noise.device)
         
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i, ...] = perturb_image(image, noise_batch[i, ...])
        
        # Early stopping 
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          losses = get_loss(self.model, image_batch[i][None, :], ref_image)
          #print(losses)
          idx = indices[bstart+i]
          margin = -(losses-curr_loss)
          heapq.heappush(priority_queue, (margin, idx))
      
      # Pick the best element and insert it into the working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        noise = self._flip_noise(noise, blocks[best_idx])
        A[best_idx] = 1
      
      # Add elements into the working set
      while len(priority_queue) > 0:
        # Pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        image_batch = perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        losses = get_loss(self.model, image_batch, ref_image)
        #print(losses)
        num_queries += 1
        margin = -(losses-curr_loss)
        
        # If the cardinality has not changed, add the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin > 0:
            break
          # Update the noise
          curr_loss = losses
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 1
          # Early stopping
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))

      priority_queue = []

      # Lazy greedy delete
      indices,  = np.where(A==1)
       
      batch_size = 100
      num_batches = int(math.ceil(len(indices)/batch_size))   
      
      for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, len(indices))
        
        image_batch = torch.zeros([bend-bstart, 3, self.width, self.height]).to(image.device)
        noise_batch = torch.zeros([bend-bstart, 3, 256, 256]).to(noise.device)
        
        for i, idx in enumerate(indices[bstart:bend]):
          noise_batch[i, ...] = self._flip_noise(noise, blocks[idx])
          image_batch[i, ...] = perturb_image(image, noise_batch[i, ...])
        
        # Early stopping
        num_queries += bend-bstart

        # Push into the priority queue
        for i in range(bend-bstart):
          losses = get_loss(self.model, image_batch[i][None, :], ref_image)
          #print(losses)
          idx = indices[bstart+i]
          margin = -(losses-curr_loss)
          heapq.heappush(priority_queue, (margin, idx))

      # Pick the best element and remove it from the working set   
      if len(priority_queue) > 0:
        best_margin, best_idx = heapq.heappop(priority_queue)
        curr_loss += best_margin
        noise = self._flip_noise(noise, blocks[best_idx])
        A[best_idx] = 0
      
      # Delete elements into the working set
      while len(priority_queue) > 0:
        # pick the best element
        cand_margin, cand_idx = heapq.heappop(priority_queue)
        
        # Re-evalulate the element
        image_batch = perturb_image(
          image, self._flip_noise(noise, blocks[cand_idx]))
        
        losses = get_loss(self.model, image_batch, ref_image)
        #print(losses)
        num_queries += 1 
        margin = -(losses-curr_loss)
      
        # If the cardinality has not changed, remove the element
        if len(priority_queue) == 0 or margin <= priority_queue[0][0]:
          # If there is no element that has negative margin, then break
          if margin >= 0:
            break
          # Update the noise
          curr_loss = losses
          noise = self._flip_noise(noise, blocks[cand_idx])
          A[cand_idx] = 0
          # Early stopping
        # If the cardinality has changed, push the element into the priority queue
        else:
          heapq.heappush(priority_queue, (margin, cand_idx))
      
      priority_queue = []
    
    return noise, num_queries, curr_loss


def split_block(upper_left, lower_right, block_size):
    """Split an image into a set of blocks. 
    Note that a block consists of [upper_left, lower_right, channel]
    
    Args:
      upper_left: [x, y], the coordinate of the upper left of an image
      lower_right: [x, y], the coordinate of the lower right of an image
      block_size: int, the size of a block

    Returns:
      blocks: list, the set of blocks
    """
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
        for c in range(3):
            blocks.append([[x, y], [x+block_size, y+block_size], c])
    return blocks

def perturb_image(image, noise):
    """Given an image and a noise, generate a perturbed image. 
    First, resize the noise with the size of the image. 
    Then, add the resized noise to the image.

    Args:
      image: torch tensor of size [3, 299, 299], an original image
      noise: torch tensor of size [3, 256, 256], a noise
      
    Returns:
      adv_iamge: torch tensor of size [3, 299, 299], an perturbed image   
    """
    adv_image = image + T.Resize(size = (image.shape[2], image.shape[3]), interpolation=T.InterpolationMode.NEAREST)(noise)
    adv_image = torch.clip(adv_image, 0., 1.)
    return adv_image
    
def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', epsilon=0.05):
    max_queries = 10000
    #epsilon = 0.01 # default 0.05
    batch_size = 64
    block_size = 32
    no_hier = False
    max_iters = 1
    model = model.to(device)
    compress_image = compress_image.to(device)
    local_search = LocalSearchHelper(model, epsilon, max_iters)

    # Local variables
    adv_image = compress_image.clone()
    num_queries = 0
    upper_left = [0, 0]
    lower_right = [256, 256]
    blocks = split_block(upper_left, lower_right, block_size)
    
    # Initialize a noise to -epsilon
    noise = -epsilon*torch.ones([3, 256, 256]).to(device)

    # Construct a batch
    num_blocks = len(blocks)
    batch_size = batch_size if batch_size > 0 else num_blocks
    curr_order = np.random.permutation(num_blocks)
    with torch.no_grad():
      # Main loop
      while True:
        # Run batch
        num_batches = int(math.ceil(num_blocks/batch_size))
        for i in range(num_batches):
          # Pick a mini-batch
          bstart = i*batch_size
          bend = min(bstart + batch_size, num_blocks)
          blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
          # Run local search algorithm on the mini-batch
          noise, queries, loss = local_search.perturb(
            compress_image, ref_image, noise, blocks_batch)
          num_queries += queries
          # If query count exceeds the maximum queries, then return False
          if num_queries > max_queries:
            return adv_image
          # Generate an adversarial image
          adv_image = perturb_image(compress_image, noise)
      
        # If block size >= 2, then split the iamge into smaller blocks and reconstruct a batch
        if not no_hier and block_size >= 2:
          block_size //= 2
          blocks = split_block(upper_left, lower_right, block_size)
          num_blocks = len(blocks)
          batch_size = batch_size if batch_size > 0 else num_blocks
          curr_order = np.random.permutation(num_blocks)
        # Otherwise, shuffle the order of the batch
        else:
          curr_order = np.random.permutation(num_blocks)


if __name__ == "__main__":
    test_main(attack)