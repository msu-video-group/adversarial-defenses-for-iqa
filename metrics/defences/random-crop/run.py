import torch
from torchvision import transforms
from defence_evaluate import test_main

class Defence:
    def __init__(self, size=128):
        self.size = int(size)

    def __call__(self, image):
        return transforms.RandomCrop(self.size)(image)

if __name__ == "__main__":
    test_main(Defence)