import torch
from torchvision import transforms
from defence_evaluate import test_main

class Defence:
    def __init__(self, angle_limit=15):
        self.angle_limit = int(angle_limit)


    def __call__(self, image):
        return transforms.RandomRotation(self.angle_limit)(image)
   
if __name__ == "__main__":
    test_main(Defence)