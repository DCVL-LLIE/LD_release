import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
import torchvision.models as models
from torchvision.models import vgg16
from math import exp

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')
    
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class KLDLoss(nn.Module):
    def __init__(self, 
                 beta_vae=True,
                 beta=1.0, 
                 start_epoch=0, 
                 end_epoch=800, 
                 kl_loss_cycle_len=0,):
        super(KLDLoss, self).__init__()
        self.beta_vae = beta_vae
        self.beta = beta  # beta coefficient for KL divergence
        self.kl_loss_anneal_end = end_epoch
        self.kl_loss_anneal_start = start_epoch
        self.kl_loss_cycle_len = kl_loss_cycle_len
        
    def forward(self, pred_dict, cur_epoch):
        stats_dict = dict()
        qm, qv = pred_dict['posterior_distrib']
        pm, pv = pred_dict['prior_distrib']
        kl_loss = self.kl_normal_mean(qm, qv, pm, pv)
        kl_stat_loss = kl_loss
        kl_loss = kl_stat_loss
        stats_dict['kl_loss'] = kl_stat_loss
        anneal_weight = 1.0

        if self.beta_vae:
            anneal_epoch = cur_epoch
            anneal_start = self.kl_loss_anneal_start
            anneal_end = self.kl_loss_anneal_end
            if self.kl_loss_cycle_len > 1:
                anneal_epoch = cur_epoch % self.kl_loss_cycle_len
                anneal_start = 0
                anneal_end = self.kl_loss_cycle_len // 2 # optimize full weight for second half of cycle
            if anneal_epoch >= anneal_start:
                anneal_weight = (anneal_epoch - anneal_start) / (anneal_end - anneal_start)
            anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight

        stats_dict['kl_anneal_weight'] = anneal_weight

        return self.beta * anneal_weight * kl_loss, stats_dict
    
    def kl_normal_mean(self, qm, qv, pm, pv):
        kl_loss = 0.5 * torch.mean(torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        return kl_loss


class PerceptualNetwork(torch.nn.Module):
    def __init__(self, loss_weight):
        super(PerceptualNetwork, self).__init__()
        self.loss_weight = loss_weight

        self.vgg_model = vgg16(pretrained=True)
        self.vgg_model = self.vgg_model.features[:16].to('cuda')
        for param in self.vgg_model.parameters():
            param.requires_grad = False
        
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))
        return (sum(loss)/len(loss)) * self.loss_weight


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

class MSSSIM(torch.nn.Module):
    def __init__(self, loss_weight=1.0, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.loss_weight = loss_weight
        
    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return (1 - msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)) * self.loss_weight
    
