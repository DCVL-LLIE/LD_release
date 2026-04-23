
from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
from thop import profile
import time 

parser = argparse.ArgumentParser(
    description='Image Enhancement using LD')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument(
    '--opt', type=str, default='Options/SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='SDSD_indoor', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--sampling', action='store_true', help='Use the sampling')
parser.add_argument('--num_samples', "-n", default=1, type=int, help='Number of samples')
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False

x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################
if args.GT_mean:
    print("Processing GT_mean")

model_restoration = create_model(opt).net_g

checkpoint = torch.load(weights)
checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'], strict=False)
except RuntimeError as e:
    print(f"RuntimeError occurred: {e}")
    new_checkpoint = {}
    for k in checkpoint['params']:
        if not k.startswith('module.'):
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        else:
            new_checkpoint[k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint, strict=False)

print("===> Testing using weights: ", weights)
model_restoration.to(DEVICE)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dummy = torch.ones([1,3,256,256]).to(DEVICE)
dummy_out = model_restoration.module.test_nonsample(dummy, use_conditional_prior=True, use_mean=True)

start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
     # Mega-paramete
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
# stx()
os.makedirs(result_dir, exist_ok=True)


##################################
def calc_metrics(restored, target):
    
    if isinstance(restored, tuple):
        restored = restored[0]  

    # Unpad images to original dimensions
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).cpu(
    ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    if args.GT_mean:
        # This test setting is the same as KinD, LLFlow, and recent diffusion models
        # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
        mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
        restored = np.clip(restored * (mean_target / mean_restored), 0, 1)
        
    sample_psnr = utils.PSNR(target, restored)
    sample_ssim = utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored))
    
    return sample_psnr, sample_ssim, restored

all_data = {}

def sampling_image(input_, target, name=None):
    #start_time = time.time()
    if args.sampling:
        pred_list, psnr_list, ssim_list =[], [], []

        start.record()
        #start_time = time.time()
        restored_list = model_restoration.module.test_sample(input_, args.num_samples)
        end.record()
        #end_time = time.time()
        for restored in restored_list:
            sample_psnr, sample_ssim, restored = calc_metrics(restored, target)

            psnr_list.append(sample_psnr)
            ssim_list.append(sample_ssim)
            pred_list.append(restored)

        index = psnr_list.index(max(psnr_list))
        
        best_psnr = psnr_list[index]
        best_ssim = ssim_list[index]
        best_pred = pred_list[index]

    else:
        start.record()
        restored = model_restoration.module.test_nonsample(input_, use_conditional_prior=True, use_mean=True)
        end.record()
        best_psnr, best_ssim, restored = calc_metrics(restored, target)
        best_pred = restored

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)  
    elapsed_time = elapsed_ms / 1000.0

    #elapsed_time = end_time - start_time
    print(elapsed_time)
    return best_psnr, best_ssim, best_pred, elapsed_time
##############################################

psnr = []
ssim = []
_time = []
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    if '~' in opt['dataroot_gt']:
        opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:
        opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            input_ = input_.cuda()
            best_psnr, best_ssim, restored, sample_time = sampling_image(input_, target, os.path.splitext(os.path.split(inp_path)[-1])[0])

            psnr.append(best_psnr)
            ssim.append(best_ssim)
            _time.append(sample_time)
            
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:

    input_dir = opt['datasets']['val']['dataroot_lq']
    target_dir = opt['datasets']['val']['dataroot_gt']
    print(input_dir)
    print(target_dir)

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

    target_paths = natsorted(glob(os.path.join(
        target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

    with torch.inference_mode():

        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):
            
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(inp_path)) / 255.
            target = np.float32(utils.load_img(tar_path)) / 255.

            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            
            best_psnr, best_ssim, restored, sample_time = sampling_image(input_, target, os.path.splitext(os.path.split(inp_path)[-1])[0])
           
            psnr.append(best_psnr)
            ssim.append(best_ssim)
            _time.append(sample_time)
            utils.save_img((os.path.join(result_dir, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
_time = np.mean(np.array(_time))

print("PSNR: %f " % (psnr))
print("SSIM: %f " % (ssim))
print("TIME: %4f " % (_time))

file_path = os.path.join(result_dir, "results.txt")

with open(file_path, "w") as file:
    file.write("PSNR: %f\n" % psnr)
    file.write("SSIM: %f\n" % ssim)
    file.write("TIME: %f\n" % _time)

