import torch
import subprocess
subprocess.run('bash ./dfsrc/setup.sh', shell=True, check=True)
from dfsrc import DiffJPEGCoding
from defence_evaluate import test_main

class Defence:
    def __init__(self, q=50):
        q = int(q)
        self.q = torch.tensor([q])

    def __call__(self, image):
        sh = image.shape
        local_q = torch.repeat_interleave(self.q, sh[0]).to(image.device)
        res = torch.nn.functional.pad(image, [0, 16 - image.shape[3] % 16, 0, 16 - image.shape[2] % 16], 'reflect')
        diff_jpeg_module = DiffJPEGCoding(ste=True)
        image_coded_diff = diff_jpeg_module(
            res*255,
            local_q,
        )
        image_coded_diff = image_coded_diff[..., :sh[2], :sh[3]]/255
        return image_coded_diff
   
if __name__ == "__main__":
    test_main(Defence)
