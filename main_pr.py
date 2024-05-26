import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_resizer import Resizer
from functools import partial

from utils.fastmri_utils import fft2c_new, ifft2c_new
from torchmetrics.image import StructuralSimilarityIndexMeasure
# >>> preds = torch.rand([3, 3, 256, 256])
# >>> target = preds * 0.75
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
import lpips
# from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

from guided_diffusion.gaussian_diffusion import _extract_into_tensor

class PhaseRetrievalOperator(torch.nn.Module):
    def __init__(self, oversample, **kwargs):
        super(PhaseRetrievalOperator,self).__init__()
        self.pad = int((oversample / 8.0) * 256)
        # self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2c_new(padded).abs() #**2
        return amplitude, self.pad

def derivative(batch):
    # pad batch
    x = torch.nn.functional.pad(batch, (1, 1, 1, 1), mode='reflect')
    dx2 = (x[..., 1:-1, :-2] - x[..., 1:-1, 2:])**2
    dy2 = (x[..., :-2, 1:-1] - x[..., 2:, 1:-1])**2
    # print(dx2.shape)
    # print(dy2.shape)
    mag = torch.sqrt(dx2 + dy2)
    thred = torch.mean(mag)
    return (mag>thred/10.0).float() #.to(torch.cuda.FloatTensor)
  
# def align(data):
#     def SSD(dat1, dat2):
#         return torch.sum((derivative(dat1)-derivative(dat2))**2)
#     out = torch.zeros_like(data)
#     out[:,0,...] = data[:,0,...]
#     util.imsave(util.tensor2uint(out[:,0:1,...]), os.path.join('x0_1.png'))
#     for i in range(1,3):
#         x0 = SSD(out[:,0:1,...],data[:,i:i+1,...])
#         x1 = SSD(out[:,0:1,...],torch.rot90(data[:,i:i+1,...],2,[2,3]))
#         if x0 <= x1:
#             out[:,i:i+1,...] = data[:,i:i+1,...]
#         else:
#             out[:,i:i+1,...] = torch.rot90(data[:,i:i+1,...],2,[2,3])
#     return out
# loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

def align(data,ref):
    # def SSD(dat1, dat2):
    #     return torch.sum((derivative(dat1)-derivative(dat2))**2)
    out_tab = []
    for i in range(2):
        Out = torch.zeros_like(data)
        if i == 0:
            Out[:,0,...] = data[:,0,...]
        else:
            Out[:,0:1,...] = torch.rot90(data[:,0:1,...],2,[2,3])
        for j in range(2):
            if j == 0:
                Out[:,1,...] = data[:,1,...]
            else:
                Out[:,1:2,...] = torch.rot90(data[:,1:2,...],2,[2,3])
            for k in range(2):
                if k == 0:
                    Out[:,2,...] = data[:,2,...]
                else:
                    Out[:,2:3,...] = torch.rot90(data[:,2:3,...],2,[2,3])
                out_tab.append(Out.clone())

    X_gray = [0.299*img[:,0:1,...] + 0.587*img[:,1:2,...] + 0.114*img[:,2:3,...] for img in out_tab]
    ref_gray = 0.299*ref[:,0:1,...] + 0.587*ref[:,1:2,...] + 0.114*ref[:,2:3,...]
    Res = [ssim(img,ref_gray).item() for img in X_gray]
    # Res = [torch.norm(img-ref).item() for img in out_tab]
    return out_tab[Res.index(max(Res))]
    # out = torch.zeros_like(data)
    # for i in range(3):
    #     norm1 = torch.norm(data[:,i:i+1,...]-ref[:,i:i+1,...])
    #     norm2 = torch.norm(torch.rot90(data[:,i:i+1,...],2,[2,3])-ref[:,i:i+1,...])
    #     if norm1<= norm2:
    #         out[:,i:i+1,...] = data[:,i:i+1,...]
    #     else:
    #         out[:,i:i+1,...] = torch.rot90(data[:,i:i+1,...],2,[2,3])
    # return out
    
    # Res = [torch.norm(x-ref).item() for x in out_tab]

    # out[:,0,...] = data[:,0,...]
    # util.imsave(util.tensor2uint(out[:,0:1,...]), os.path.join('x0_1.png'))
    # for i in range(1,3):
    #     x0 = SSD(out[:,0:1,...],data[:,i:i+1,...])
    #     x1 = SSD(out[:,0:1,...],torch.rot90(data[:,i:i+1,...],2,[2,3]))
    #     if x0 <= x1:
    #         out[:,i:i+1,...] = data[:,i:i+1,...]
    #     else:
    #         out[:,i:i+1,...] = torch.rot90(data[:,i:i+1,...],2,[2,3])
    # return out

    # loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    # lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
    #                     lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]

# def align(data,ref):
#     # def SSD(dat1, dat2):
#     #     return torch.sum((derivative(dat1)-derivative(dat2))**2)
#     out_tab = []
#     for i in range(4):
#         Out = torch.zeros_like(data)
#         if i == 0:
#             Out[:,0,...] = data[:,0,...]
#         elif i == 1:
#             Out[:,0:1,...] = torch.rot90(data[:,0:1,...],2,[2,3])
#         elif i == 2:
#             Out[:,0,...] = -data[:,0,...]
#         elif i == 3:
#             Out[:,0:1,...] = -torch.rot90(data[:,0:1,...],2,[2,3])
#         for j in range(4):
#             if j == 0:
#                 Out[:,1,...] = data[:,1,...]
#             elif j == 1:
#                 Out[:,1:2,...] = torch.rot90(data[:,1:2,...],2,[2,3])
#             elif j == 2:
#                 Out[:,1,...] = -data[:,1,...]
#             elif j == 3:
#                 Out[:,1:2,...] = -torch.rot90(data[:,1:2,...],2,[2,3])
#             for k in range(4):
#                 if k == 0:
#                     Out[:,2,...] = data[:,2,...]
#                 elif k == 1:
#                     Out[:,2:3,...] = torch.rot90(data[:,2:3,...],2,[2,3])
#                 elif k == 2:
#                     Out[:,2,...] = -data[:,2,...]
#                 elif k == 3:
#                     Out[:,2:3,...] = -torch.rot90(data[:,2:3,...],2,[2,3])
#                 out_tab.append(Out.clone())

#     X_gray = [0.299*img[:,0:1,...] + 0.587*img[:,1:2,...] + 0.114*img[:,2:3,...] for img in out_tab]
#     ref_gray = 0.299*ref[:,0:1,...] + 0.587*ref[:,1:2,...] + 0.114*ref[:,2:3,...]
#     # Res = [ssim(img,ref_gray).item() for img in X_gray]
#     Res = [torch.norm(img-ref).item() for img in out_tab]
#     return out_tab[Res.index(max(Res))]

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img         = 12.75/255.0       # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'diffusion_ffhq_10m'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusino model
    testset_name            = 'demo_test' #'demo_test'    # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps     = 1000 #1000
    iter_num                = 1000                # set number of sampling iterations
    skip                    = num_train_timesteps//iter_num     # skip interval
    sr_mode                 = 'blur'            # 'blur', 'cubic' mode of sr up/down sampling
    lambda_                 = 1.                # key parameter lambda
    ddim_sample             = False             # sampling method
    generate_mode           = 'DiffPIR'         # DiffPIR; DPS; vanilla
    skip_type               = 'quad'            # uniform, quad
    eta                     = 0.                # eta for ddim sampling
    zeta                    = 0.1               
    guidance_scale          = 1.0   

    test_sf                 = [4]               # set scale factor, default: [2, 3, 4], [2], [3], [4]
    inIter                  = 1                 # iter num for sr solution: 4-6
    gamma                   = 1/100             # coef for iterative sr solver 20steps: 0.05-0.10 for zeta=1, 0.09-0.13 for zeta=0 
    task_current            = 'sr'              # 'sr' for super resolution
    n_channels              = 3                 # fixed
    cwd                     = '' 
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    results                 = os.path.join(cwd, 'results_pr_noise_1000_ffhq_0.05_with_dm_raar')      # fixed
    result_name             = f'{testset_name}_{task_current}_{generate_mode}_{sr_mode}{str(test_sf)}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # noise schedule 
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0

    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
    t_start                 = num_train_timesteps - 1   

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    if generate_mode != 'DPS_y0':
        # for DPS_yt, we can avoid backward through the model
        for k, v in model.named_parameters():
            v.requires_grad = False
    model = model.to(device)

    # # some code to change the generative model for gray-scale image
    # model.input_blocks[0][0].in_channels = 1
    # model.input_blocks[0][0].weight = torch.nn.Parameter(model.input_blocks[0][0].weight.sum(dim=1,keepdim=True))

    logger.info('model_name:{}, sr_mode:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, sr_mode, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('analytic iter num:{}, gamma:{}'.format(inIter, gamma))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    
    test_results_ave = OrderedDict()

    pr = PhaseRetrievalOperator(oversample=4.0)
    # get the images and run fft to run HIO method

    for sub in ['deg','recon']:
        if not os.path.exists(os.path.join(E_path,sub)):
            os.makedirs(os.path.join(E_path,sub),exist_ok=True)
    for f_path in L_paths:
        img_name, ext = os.path.splitext(os.path.basename(f_path))
        img_H = util.imread_uint(f_path, n_channels=n_channels)
        img = util.uint2tensor4(img_H).cuda() #*255.0
        meas, pad = pr(img)
        meas = meas + 0.05*torch.randn_like(meas)
        util.imsave(util.tensor2uint(torch.clamp((meas+1.0)/2.0,0,1)), os.path.join(E_path, 'deg',img_name + '_'+'measurement_x'+'.png'))

        # define the two projections
        def P_m(x,meas):
            out = fft2c_new(x)
            phase = out/out.abs()
            out2 = meas*phase
            out3 = ifft2c_new(out2)
            return out3[...,0]
        def P_i(data, pad):
            out = torch.zeros_like(data)
            out[:,:,pad:-pad,pad:-pad] = data[:,:,pad:-pad,pad:-pad]
            return out
        
        for i_run in range(20):

            os.makedirs(os.path.join(E_path,sub,str(i_run)),exist_ok=True)
            x = torch.randn(1,3,256,256).cuda()
            x = torch.randn_like(x)
            # --------------------------------
            # (4) main iterations
            # --------------------------------
            X0 = torch.randn_like(meas)
            progress_img = []
            # create sequence of timestep for sampling
            skip = num_train_timesteps//iter_num
            if skip_type == 'uniform':
                seq = [i*skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps-1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::max(len(seq)//10,1)]
            if progress_seq[-1] != seq[-1]:
                progress_seq.append(seq[-1])
            
            # reverse diffusion for one image from random noise
            print(img_name)
            for i in list(range(num_train_timesteps))[::-1]:

                sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
                sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
                reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

                t_step = i
                vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)
                if not ddim_sample:
                    out = diffusion.p_sample(
                        model,
                        x,
                        vec_t,
                        clip_denoised=True,
                        denoised_fn=None,
                        cond_fn=None
                    )
                    # here update the estimated x_0
                    # X0 = out["pred_xstart"]
                    
                    # out["pred_xstart"] = 0.8*out["pred_xstart"] + 0.2*X0[:,:,pad:-pad,pad:-pad]
                    # replace the iteration
                    model_mean, _, _ = diffusion.q_posterior_mean_variance(
                        x_start=out["pred_xstart"], x_t=x, t=vec_t)
                    noise = torch.randn_like(x)
                    nonzero_mask = (
                        (vec_t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                    )  # no noise when t == 0
                    out["sample"] = model_mean + nonzero_mask * torch.exp(0.5 * _extract_into_tensor(diffusion.posterior_log_variance_clipped, vec_t, x.shape)) * noise
                    beta = diffusion.betas[t_step]
                    alpha_prod_t = alphas_cumprod[t_step]
                    sigma_t_s = (1-diffusion.alphas_cumprod[t_step])
                    lamb = beta/(np.sqrt(1-beta)*0.1*sigma_t_s)
                    lamb = np.sqrt(alpha_prod_t)
                    # lamb = alpha_prod_t
                    X0[:,:,pad:-pad,pad:-pad] = (out["pred_xstart"]+1.0)/2.0
                    # for iner in range(4):
                    Y0 = P_m(X0,meas)
                        # print(Y0.shape)
                    O1 = P_i(2*Y0-X0,pad)
                    Z1 = torch.zeros_like(X0)
                    # Z1[:,:,pad:-pad,pad:-pad] = out["pred_xstart"]
                    # O3 = P_i(X0,pad)
                    # X0 = 0.45*(O1 + X0 - Y0) + 0.55*Z1
                    X0 = O1 + X0 - Y0
                    # noise case
                    X0 = 0.75*X0 + (1-0.75)*P_m(X0,meas) # noiseless
                    # X0 = 0.95*X0 + (1-0.95)*P_m(X0,meas) # 0.01
                    X0 = torch.clamp((X0),0,1)

                    out_tmp = (1-lamb)*out["pred_xstart"] + lamb*(2*X0[:,:,pad:-pad,pad:-pad]-1)
                    # add back noise to get the 
                    out["sample"] = np.sqrt(alpha_prod_t)*out_tmp+ nonzero_mask * torch.exp(0.5 * _extract_into_tensor(diffusion.posterior_log_variance_clipped, vec_t, x.shape)) * noise
                else:
                    out = diffusion.ddim_sample(
                        model,
                        x,
                        vec_t,
                        clip_denoised=True,
                        denoised_fn=None,
                        cond_fn=None,
                        eta=0,
                    )

                
                x = out["sample"]
                x0 = out["pred_xstart"]

            util.imsave(util.tensor2uint(X0[:,:,pad:-pad,pad:-pad]), os.path.join(E_path, 'recon',str(i_run),img_name + '.png'))
        
if __name__ == '__main__':

    main()
