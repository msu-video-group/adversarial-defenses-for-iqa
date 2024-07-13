import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from defence_evaluate import test_main

def unsharp_masking(images, kernel_size=5, sigma=1, amount=1.0):
    """
    Applies unsharp masking to a batch of images.

    Args:
        images (torch.Tensor): Input images with shape (batch_size, channels, height, width).
        kernel_size (int): Size of the Gaussian kernel. Default is 5.

    Returns:
        torch.Tensor: Sharpened images with the same shape as the input.
    """

    # Apply Gaussian blur to the input images
    blurred = GaussianBlur(kernel_size, sigma)(images)

    # Compute the sharpened images
    sharpened = images + (images - blurred) * amount

    return torch.clip(sharpened, 0, 1)

class Defence:
    def __init__(self, kernel_size=5, sigma=1, amount=1):
        self.kernel_size=int(kernel_size)
        self.amount=int(amount)
        self.sigma = float(sigma)

    def __call__(self, image):
        return unsharp_masking(image, self.kernel_size, self.sigma, self.amount)
   
if __name__ == "__main__":
    test_main(Defence)