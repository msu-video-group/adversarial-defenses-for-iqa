import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize

torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
METRIC_MAX_VAL = 100.0


def flow_st(images, flows, device='cpu'):
    images_shape = images.size()
    flows_shape = flows.size()
    batch_size = images_shape[0]
    H = images_shape[2]
    W = images_shape[3]
    basegrid = torch.stack(torch.meshgrid(torch.arange(0,H), torch.arange(0,W))) #(2,H,W)
    sampling_grid = basegrid.unsqueeze(0).type(torch.float32).to(device) + flows.to(device)
    sampling_grid_x = torch.clamp(sampling_grid[:,1],0.0,W-1.0).type(torch.float32)
    sampling_grid_y = torch.clamp(sampling_grid[:,0],0.0,H-1.0).type(torch.float32)

    x0 = torch.floor(sampling_grid_x).type(torch.int64)
    x1 = x0 + 1
    y0 = torch.floor(sampling_grid_y).type(torch.int64)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 2)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 2)
    y1 = torch.clamp(y1, 0, H - 1)

    Ia = images[:,:,y0[0,:,:], x0[0,:,:]]
    Ib = images[:,:,y1[0,:,:], x0[0,:,:]]
    Ic = images[:,:,y0[0,:,:], x1[0,:,:]]
    Id = images[:,:,y1[0,:,:], x1[0,:,:]]

    x0 = x0.type(torch.float32)
    x1 = x1.type(torch.float32)
    y0 = y0.type(torch.float32)
    y1 = y1.type(torch.float32)

    wa = (x1 - sampling_grid_x) * (y1 - sampling_grid_y)
    wb = (x1 - sampling_grid_x) * (sampling_grid_y - y0)
    wc = (sampling_grid_x - x0) * (y1 - sampling_grid_y)
    wd = (sampling_grid_x - x0) * (sampling_grid_y - y0)
    perturbed_image = wa.unsqueeze(0)*Ia+wb.unsqueeze(0)*Ib+wc.unsqueeze(0)*Ic+wd.unsqueeze(0)*Id

    return perturbed_image.type(torch.float32).to(device)

def flow_loss(flows, padding_mode='constant', epsilon=1e-8, device='cpu'):
    paddings = (1,1,1,1)
    padded_flows = F.pad(flows,paddings,mode=padding_mode,value=0)
    shifted_flows = [
    padded_flows[:, :, 2:, 2:],  # bottom right (+1,+1)
    padded_flows[:, :, 2:, :-2],  # bottom left (+1,-1)
    padded_flows[:, :, :-2, 2:],  # top right (-1,+1)
    padded_flows[:, :, :-2, :-2]  # top left (-1,-1)
    ]
    #||\Delta u^{(p)} - \Delta u^{(q)}||_2^2 + # ||\Delta v^{(p)} - \Delta v^{(q)}||_2^2 
    loss=0
    for shifted_flow in shifted_flows:
        loss += torch.sum(torch.square(flows[:, 1] - shifted_flow[:, 1]) + torch.square(flows[:, 0] - shifted_flow[:, 0]) + epsilon).to(device)
    return loss.type(torch.float32)

def metric_d(model, x0, x1=None, ref_img=None, maxval=METRIC_MAX_VAL):
    #d0 = model.forward(x0)
    d0 = model.forward(ref_img, x0) if ref_img is not None else model.forward(x0)
    if x1 is None:
        return d0, maxval * 1.5 # metric-dependent const
    #d1 = model.forward(x1)
    d1 = model.forward(ref_img, x1) if ref_img is not None else model.forward(x1)
    return d0, d1

def rank_loss(s_adv, s_other):
    #print('iter')
    return (s_other/(s_adv+s_other)) #.float().clone()

def func(flows, model, x_prey, x_other, alpha=50, beta=1e-5, maxval=METRIC_MAX_VAL, device='cpu'): # default: alpha=50, beta=0.05
    
    flows = torch.from_numpy(flows).view((1,2,)+x_prey.size()[2:]).to(device)
    flows.requires_grad=True
    pert_out = flow_st(x_prey, flows, device)
    
    s_adv, s_other = metric_d(model, pert_out, x_other, maxval)
    
    L_adv = rank_loss(s_adv, s_other)
    
    L_flow = flow_loss(flows, device=device)
    
    L_final = alpha * L_adv + beta * L_flow
    
    model.zero_grad()
    L_final.backward()
    
    gradient = flows.grad.data.view(-1).detach().cpu().numpy()
    
    if s_adv > s_other:
        return 0, gradient
    
    return L_final.item(), gradient


### Scipy's LBFGS, slower than Torch
# def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu'):
#     MAXITER = 3
#     ALPHA = 50
#     BETA = 1e-5
#     input_prey = compress_image.clone().to(device)
#     init_flows = np.zeros((1,2,)+compress_image.size()[2:]).reshape(-1)
#     results = optimize.fmin_l_bfgs_b(func, init_flows, args=(model, compress_image.to(device), None, ALPHA, BETA, METRIC_MAX_VAL, device), maxiter=MAXITER, disp=False)
#     flows = torch.from_numpy(results[0]).view((1,2,)+input_prey.size()[2:])    
#     x_adv = flow_st(input_prey, flows, device)
#     return x_adv.detach().cpu()




def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', lr=2.5):
    #LR = 0.1 # default 2.5
    MAXITER = 3
    ALPHA = 50
    BETA = 1e-5
    input_prey = compress_image.clone().to(device)
    flows = np.zeros((1,2,) + compress_image.size()[2:])
    #print(flows.shape)
    flows = torch.from_numpy(flows)
    flows = torch.autograd.Variable(flows).to(device)
    flows.requires_grad = True
    opt = torch.optim.LBFGS([flows], lr=lr, max_iter=1, max_eval=25)

    def closure():
        opt.zero_grad()

        pert_out = flow_st(input_prey, flows, device)
        
        s_adv, s_other = metric_d(model, pert_out, None, ref_img=ref_image, maxval=metric_range)
        if model.lower_better:
            s_adv = metric_range - s_adv
            s_other = metric_range * 0.1 # adjustable loss parameter. For NR metrics == 1.5 * metric_range, for FR - 0.1 * metric_range.
        L_adv = rank_loss(s_adv, s_other)
        
        L_flow = flow_loss(flows, device=device)
        
        L_final = ALPHA * L_adv + BETA * L_flow
    
        model.zero_grad()
        L_final.backward()
        return L_final
    

    for i in range(MAXITER):
        opt.step(closure)
        
    #results = optimize.fmin_l_bfgs_b(func, init_flows, args=(model, compress_image.to(device), None, ALPHA, BETA, METRIC_MAX_VAL, device), maxiter=MAXITER, disp=False)
    #flows = torch.from_numpy(results[0]).view((1,2,)+input_prey.size()[2:])    
    x_adv = flow_st(compress_image.clone().to(device), flows, device)
    return x_adv.detach().cpu()


from fgsm_evaluate import test_main


if __name__ == "__main__":
    test_main(attack)
