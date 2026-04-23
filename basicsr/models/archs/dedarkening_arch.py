import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2
import numpy as np

import importlib
from basicsr.models.archs.networks.networks import Unet
from basicsr.models.archs.layers.transformer import FormerGaussian

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

class LLUNet(nn.Module):
    def __init__(self,
                 ## Transformer
                 in_dim=3,
                 out_dim=3,
                 dim=40,
                 level=2,
                 num_blocks=[2,2,2],
                 z_channels=10,
                 ):
        
        super(LLUNet, self).__init__()

        self.latent_size = z_channels
        
        self.unet = Unet(in_dim=in_dim, out_dim=out_dim, dim=dim, 
                         level=level, num_blocks=num_blocks, 
                         z_channels=z_channels)

        self.p_net = FormerGaussian(dim=dim, level=level, num_blocks=num_blocks, 
                                    in_ch=in_dim, z_channels=z_channels, use_feature=False)
            
        self.q_net = FormerGaussian(dim=dim, level=level, num_blocks=num_blocks, 
                                    in_ch=in_dim*2, z_channels=z_channels, use_feature=False)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_l, x_n):
        enhance, x_pred_dict = self.single_step(x_l, x_n)

        return enhance, x_pred_dict
    
    def single_step(self, x_l, x_n):
        B = x_l.size(0)

        skip_connection = x_l

        qm, qv = self.posterior(x_l, x_n)
        z = self.rsample(qm, qv)

        pm, pv = self.prior(x_l)

        darkness = self.decode(x_l, z)

        enhance = skip_connection + darkness

        x_pred_dict = {'decoder_out' : enhance}
        x_pred_dict['darkness_out'] = darkness
        x_pred_dict['posterior_distrib'] = (qm, qv)
        x_pred_dict['prior_distrib'] = (pm, pv)

        return enhance, x_pred_dict
    
    def prior(self, x_l):
        '''
        Input:
        - x_l       (B x C x H x W)

        Returns:
        - mean, var (B x latent_size)
        '''

        prior_out = self.p_net(x_l)
        mean = prior_out[:, :self.latent_size]
        logvar = prior_out[:, self.latent_size:]
        var = torch.exp(logvar)
        
        return mean, var

    def posterior(self, x_l, x_n):
        '''
        Input:
        - x_l       (B x C x H x W)
        - x_n       (B x C x H x W)

        Returns:
        - mean, var (B x latent_size)
        '''

        assert x_l.size(0) == x_n.size(0), "Batch sizes of x_l and x_n must match"
        encoder_in = torch.cat([x_l, x_n], axis=1)
        encoder_out = self.q_net(encoder_in)

        mean = encoder_out[:, :self.latent_size]
        logvar = encoder_out[:, self.latent_size:]

        var = torch.exp(logvar)

        return mean, var

    def rsample(self, mu, var):
        '''
        Return gaussian sample of (mu, var) using reparameterization trick.
        '''
        eps = torch.randn_like(mu)
        z = mu + eps*torch.sqrt(var)
        return z 
    
    def decode(self, x_l, z):
        '''
        Input:
        - z         (Bxlatent_size)
        - x_l       (BxCxHxW)

        Returns:
        - decoder_out 
        '''            
        unet_feature = self.unet(x_l, z)
        return unet_feature
    
    def test_sample(self, x_l, num_samples):
        B = x_l.size(0)
        pm, pv = self.prior(x_l)
        #pm, pv = torch.zeros((B, self.latent_size)).to(x_l), torch.ones((B, self.latent_size)).to(x_l)
        enhances , darkness = self.unet.sampling(x_l, pm, pv, num_samples)

        return enhances, darkness
    
    def test_nonsample(self, x_l, use_conditional_prior=True, use_mean=True, dist=False):
        skip_connection = x_l

        B = x_l.size(0)
        
        pm, pv = None, None
        # prior
        if use_conditional_prior:
            pm, pv = self.prior(x_l)
        else:
            # use standard normal
            pm, pv = torch.zeros((B, self.latent_size)).to(x_l), torch.ones((B, self.latent_size)).to(x_l)

        if not use_mean:
            pz = self.rsample(pm, pv)
        else:
            pz = pm # NOTE: use mean
       
        darkness = self.decode(x_l, pz)

        enhance = skip_connection + darkness

        if dist:
            dist_dict = {'distribution' : pz}
            dist_dict['mean'] = pm
            dist_dict['var'] = pv
            return enhance, dist_dict
        else:
            return enhance
    
    def test_tsne(self, x_l, x_n, use_conditional_prior=True, use_mean=True, dist=False):
        skip_connection = x_l

        qm, qv = self.posterior(x_l, x_n)

        B = x_l.size(0)
        
        pm, pv = None, None
        # prior
        pm, pv = self.prior(x_l)

        if not use_mean:
            pz = self.rsample(pm, pv)
            qz = self.rsample(qm, qv)
        else:
            pz = pm # NOTE: use mean
            qz = qm
       
        darkness = self.decode(x_l, pz)

        enhance = skip_connection + darkness

        if dist:
            dist_dict = {'prior_z' : pz}      
            dist_dict['posterior_z'] = qz    
            dist_dict['prior_mean'] = pm
            dist_dict['prior_var'] = pv
            return enhance, dist_dict
        else:
            return enhance

    def dict_out(self, x_l, x_n):
        pm, pv = self.prior(x_l)
        qm, qv = self.posterior(x_l, x_n)

        dist_dict = {'prior_mean' : pm}
        dist_dict['prior_var'] = pv
        dist_dict['posterior_mean'] = qm
        dist_dict['posterior_var'] = qv

        return dist_dict
    
class FLOPs(nn.Module):
    def __init__(self,
                 ## Transformer
                 in_dim=3,
                 out_dim=3,
                 dim=40,
                 level=2,
                 num_blocks=[2,2,2],
                 z_channels=10,
                 ):
        super(FLOPs, self).__init__()
        self.net = LLUNet(in_dim=in_dim,
                           out_dim=out_dim,
                           dim=dim,
                           level=level,
                           num_blocks=num_blocks,
                           z_channels=z_channels)
        
    def forward(self, dummy):
        out = self.net.test_nonsample(dummy, use_conditional_prior=True, use_mean=True, dist=False)
        #out = self.net.test_sample(dummy, 90)
        return out
    
if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = FLOPs(in_dim= 3,
                            out_dim= 3,
                            dim= 40,
                            level= 1,
                            num_blocks= [2,4],
                            z_channels= 256).cuda()
    model.eval()
    
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    with torch.no_grad():
        flops = FlopCountAnalysis(model,inputs)
        n_param = sum([p.nelement() for p in model.parameters()])  
        print(f'GFLOPs:{flops.total()/(1024*1024*1024)}')
        print(f'Params:{n_param}')

