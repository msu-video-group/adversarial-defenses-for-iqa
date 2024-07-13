import torch
import torch.nn as nn

from runpy import run_path
from collections import OrderedDict

import torch.nn.functional as F
import subprocess
from defence_evaluate import test_main
import argparse
import yaml
import os
from dfsrc.robustbench.model_zoo.defense import inr

class Defence:
    def __init__(self):
        subprocess.run('bash ./dfsrc/setup.sh', shell=True, check=True)
        h, w = 299, 299
        self.defence_model = inr.INR('cuda', ['disco_pgd.pth'], height=299, width=299)

    def __call__(self, image):
        res = self.defence_model.forward(image)
        return res.clamp(0.0, 1.0)
   
if __name__ == "__main__":
    test_main(Defence)