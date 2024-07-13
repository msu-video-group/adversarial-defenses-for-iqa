import torch
from defence_evaluate import test_main

class Defence:
    def __init__(self, axes=[2, 3]):
        self.axes = axes

    def __call__(self, image):
        return torch.flip(image, self.axes)
   
if __name__ == "__main__":
    test_main(Defence)
