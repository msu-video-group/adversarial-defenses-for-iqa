from pathlib import Path
import os
import sys
sys.path.append(os.path.join(Path(__file__).parent, "FPR_IQA/FPR_NI/src")) 
import torch

from model import IQANet
sys.path.remove(os.path.join(Path(__file__).parent, "FPR_IQA/FPR_NI/src")) 


class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path):
        super().__init__()
        self.device = device
        model = IQANet(weighted=True).to(device)
        
        model.load_state_dict(torch.load(model_path)['state_dict'])
        model.eval().to(device)
        
        self.model = model
        self.lower_better = False
    
    def forward(self, image, inference=False):
        
        
        patch_size = 64
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).reshape(image.shape[0], -1, 3, patch_size, patch_size)
        patches = patches.to(self.device)
        torch.backends.cudnn.enabled = False
        out = self.model(
            patches, patches
        ).mean()
        torch.backends.cudnn.enabled = True
        if inference:
            return out.detach().cpu().numpy()[0][0].item()
        else:
            return out
        
        