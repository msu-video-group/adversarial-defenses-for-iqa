import os
import torch
import pyiqa

class MetricModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = pyiqa.create_metric(
            'paq2piq',
            as_loss=True,
            device=device
        )
        self.lower_better = self.model.lower_better
        self.full_reference = False

    def forward(self, dist, inference=False):
        return self.model(dist)