import torch
from torchvision import transforms
from defence_evaluate import test_main
import numpy as np

class Defence:
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, image):
        return torch.clamp(image + self.sigma*torch.randn(image.shape, device=image.device), 0, 1)

if __name__ == "__main__":
    test_main(Defence)
