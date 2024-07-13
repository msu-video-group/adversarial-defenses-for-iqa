import torch
from defence_evaluate import test_main

class Defence:
    def __init__(self, npp=16):
        self.npp = int(npp)

    def __call__(self, image):
        npp_int = self.npp - 1
            
        x_int = torch.round(image * npp_int)
        x_float = x_int / npp_int
        return x_float
   
if __name__ == "__main__":
    test_main(Defence)
