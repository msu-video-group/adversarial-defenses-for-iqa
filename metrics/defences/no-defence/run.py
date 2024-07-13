import torch
from torchvision import transforms
from defence_evaluate import test_main

class Defence:
    def __init__(self):
        pass

    def __call__(self, image):
        return image

if __name__ == "__main__":
    test_main(Defence)