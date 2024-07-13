import torch

from torch.autograd import Variable
from fgsm_evaluate import test_main 

from IQA_pytorch import DISTS


def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', lr=0.005):
    #lr = 0.00005  # default 0.005
    loss_f = DISTS().to(device)
    iters = 10
    compress_image = Variable(compress_image.clone().to(device), requires_grad=True)
    in_image = Variable(compress_image.clone().to(device), requires_grad=False)
    optimizer = torch.optim.Adam([compress_image], lr=lr)
    sign = -1 if model.lower_better else 1
    for i in range(iters):
        score = model(ref_image.to(device), compress_image.to(device)) if ref_image is not None else model(compress_image.to(device))
        loss = loss_f(compress_image, in_image, as_loss=True).to(device) - score.to(device) * sign / metric_range
        loss.backward() 
        compress_image.grad.data[torch.isnan(compress_image.grad.data)] = 0
        optimizer.step()
        compress_image.data.clamp_(min=0, max=1)
        compress_image.data[torch.isnan(compress_image.data)] = 0
        optimizer.zero_grad() 

    res_image = (compress_image).data.clamp_(min=0, max=1)
    return res_image
            

if __name__ == "__main__":
    test_main(attack)

