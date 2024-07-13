import torch
from torchvision import transforms
from defence_evaluate import test_main

class Defence:
    def __init__(self, scale=0.5, mode='bilinear'):
        self.scale = float(scale)
        self.mode = mode

    def __call__(self, image):
        new_size = (torch.tensor(image.shape[-2:]) * self.scale)
        image = transforms.Resize((int(new_size[0]), int(new_size[1])))(image)
        resized_image = torch.nn.Upsample(size=(384, 512), mode=self.mode)(image)
        return resized_image

if __name__ == "__main__":
    test_main(Defence)
