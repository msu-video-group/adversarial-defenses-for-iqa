import torch
from torchvision.transforms import GaussianBlur
from defence_evaluate import test_main

class Defence:
    def __init__(self, kernel_size=3):
        kernel_size = int(kernel_size)
        self.kernel_size = kernel_size
        self.transform = GaussianBlur(kernel_size, 0.3*((kernel_size-1)*0.5 - 1) + 0.8)

    def __call__(self, image):
        image = self.transform(image)
        return image
   
if __name__ == "__main__":
    test_main(Defence)