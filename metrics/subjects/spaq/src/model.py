import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image



class Image_load(object):
    def __init__(self, size, stride, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.stride = stride
        self.interpolation = interpolation

    def __call__(self, img):
        image = self.adaptive_resize(img)
        return self.generate_patches(image, input_size=self.stride)

    def adaptive_resize(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size
        if h < self.size or w < self.size:
            return transforms.ToTensor()(img)
        else:
            return transforms.ToTensor()(transforms.Resize(self.size, self.interpolation)(img))

    def to_numpy(self, image):
        p = image.numpy()
        return p.transpose((1, 2, 0))

    def generate_patches(self, image, input_size, type=np.float32):
        img = self.to_numpy(image)
        img_shape = img.shape
        img = img.astype(dtype=type)
        if len(img_shape) == 2:
            H, W, = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                         for hId in hIdx
                         for wId in wIdx]
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor.squeeze(0)

    def generate_patches_custom(self, image, input_size, type=np.float32):
        # img = self.to_numpy(image)
        img = image
        img = img.to(torch.float32)
        img = img.squeeze(0)
        img_shape = img.shape
        ch, H, W = img_shape
        # if len(img_shape) == 2:
        #    H, W, = img_shape
        #    ch = 1
        # else:
        #    H, W, ch = img_shape
        # if ch == 1:
        #    img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_tensor = [img[:, hId:hId + input_size, wId:wId + input_size]
                          for hId in hIdx
                          for wId in wIdx]
        # patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor.squeeze(0)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

    def forward(self, x):
        result = self.backbone(x)
        return result


class SPAQ(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        curdir = os.path.abspath(os.path.join(__file__, os.pardir))
        model = Baseline()
        model.load_state_dict(torch.load(os.path.join(curdir, '../BL_release.pt'), map_location='cpu')['state_dict'])
        self.model = model.eval().to(device)
        self.loader = Image_load(size=512, stride=224)
    
    def forward(self, tensor):
        tensor = transforms.Resize(512)(tensor)
        patches = self.loader.generate_patches_custom(tensor, 224).to(self.device)
        return self.model(patches).mean()
    
curdir = os.path.abspath(os.path.join(__file__, os.pardir))

class MetricModel(torch.nn.Module):
    def __init__(self, device, model_path=f'{curdir}/../BL_release.pt'):
        super().__init__()
        self.device = device
        self.lower_better = False
        self.full_reference = False

        model = Baseline()
        model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
        self.model = model.eval().to(device)
        self.loader = Image_load(size=512, stride=224)
    
    def forward(self, image, inference=False):
        patches = self.loader.generate_patches_custom(image, 224).to(self.device)
        return self.model(patches).mean()
