import random
import numpy as np
import os
import argparse
from tqdm import tqdm
import utils

import torch.nn as nn
import torch
import torch.nn.functional as F

from natsort import natsorted
from glob import glob

import os
import torchvision.transforms as transforms
from PIL import Image
from skimage import img_as_ubyte

import torch.utils.data
from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse
#from basicsr.metrics.niqe import calculate_niqe
from basicsr.metrics_ours.niqe_utils import *
import torch.nn as nn

import time 

class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, low_img_dir):
        self.low_img_dir = low_img_dir
        self.train_low_data_names = []

        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):
        low = self.load_images_transform(self.train_low_data_names[index])

        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]
        return torch.from_numpy(low), img_name

    def __len__(self):
        return self.count
    
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
parser.add_argument('--sampling', action='store_true', help='Use the sampling')
parser.add_argument('--num_samples', "-n", default=1, type=int, help='Number of samples')
args = parser.parse_args()

# save evaluate image
def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

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
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 生成输出结果的文件
factor = 16
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
# stx()
os.makedirs(result_dir, exist_ok=True)

##################################
def calc_metrics(restored, h, w):
    
    if isinstance(restored, tuple):
        restored = restored[0]  

    # Unpad images to original dimensions
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).cpu(
    ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    sample_niqe = calculate_niqe(img_as_ubyte(restored), crop_border=0)

    return sample_niqe, restored

def sampling_image(input_, h, w):
    start_time = time.time()
    if args.sampling:
        pred_list, niqe_list =[], []

        restored_list = model_restoration.module.test_sample(input_, args.num_samples)
        for restored in restored_list:
            sample_niqe, restored = calc_metrics(restored, h, w)

            niqe_list.append(sample_niqe)
            pred_list.append(restored)

        index = niqe_list.index(min(niqe_list))
        best_niqe = niqe_list[index]
        best_pred = pred_list[index]
        
    else:
        if h < 1200 and w < 1200:
            restored = model_restoration.module.test_nonsample(input_, use_conditional_prior=True, use_mean=True)
        else:
            # split and test into 4 parts
            input_top_left = input_[:, :, 0::2, 0::2] 
            input_top_right = input_[:, :, 0::2, 1::2]  
            input_bottom_left = input_[:, :, 1::2, 0::2]  
            input_bottom_right = input_[:, :, 1::2, 1::2] 

            restored_top_left = model_restoration.module.test_nonsample(input_top_left, use_conditional_prior=True, use_mean=True)
            restored_top_right = model_restoration.module.test_nonsample(input_top_right, use_conditional_prior=True, use_mean=True)
            restored_bottom_left = model_restoration.module.test_nonsample(input_bottom_left, use_conditional_prior=True, use_mean=True)
            restored_bottom_right = model_restoration.module.test_nonsample(input_bottom_right, use_conditional_prior=True, use_mean=True)

            restored = torch.zeros_like(input_)
            restored[:, :, 0::2, 0::2] = restored_top_left
            restored[:, :, 0::2, 1::2] = restored_top_right
            restored[:, :, 1::2, 0::2] = restored_bottom_left
            restored[:, :, 1::2, 1::2] = restored_bottom_right

        best_niqe, restored = calc_metrics(restored, h, w)
        best_pred = restored

    end_time = time.time()
    elapsed_time = end_time - start_time

    return best_niqe, best_pred, elapsed_time

niqe = []
_time = []

low_data_names = opt['datasets']['val']['dataroot_unpaired']
TestDataset = MemoryFriendlyLoader(low_img_dir=low_data_names)

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0, shuffle=True)

def save_images(tensor, original_path, save_dir):
    """
    Saves the image tensor to the specified directory with the original file extension.
    Args:
        tensor (torch.Tensor): Image tensor to save.
        original_path (str): The original file path to extract the name and extension.
        save_dir (str): The directory where the image will be saved.
    """
    # Extract the original file name and extension
    file_name = os.path.basename(original_path)
    save_path = os.path.join(save_dir, file_name)
    
    # Convert tensor to image
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    # Save the image
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(save_path)

with torch.inference_mode():
    for inputs, name in tqdm(test_queue, unit='batch'):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()    

            #image_name = name
            #image_name = image_name[0].split('/')[-1].split('.')[0]
            #save_path = os.path.join(result_dir, image_name + '.png')
            input_ = inputs.cuda()

            original_path = name[0]
            file_name = os.path.basename(original_path)
            save_path = os.path.join(result_dir, file_name)

            # Padding in case images are not multiples of 4

            h, w = input_.shape[2], input_.shape[3]
            print(h,w)
            if h < 1200 and w < 1200:
                factor = 4
            else:
                factor = 8
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            best_niqe, restored, sample_time = sampling_image(input_, h, w)
            #print(restored.shape)
            utils.save_img(save_path, img_as_ubyte(restored))
            #save_images(torch.tensor(restored).unsqueeze(0), original_path, save_dir)
            _time.append(sample_time)
            niqe.append(best_niqe)

niqe = np.mean(np.array(niqe))
_time = np.mean(np.array(_time))
print("NIQE: %f " % (niqe))
print("TIME: %f " % (_time))

file_path = os.path.join(result_dir, "results.txt")
# 결과를 텍스트 파일로 저장
with open(file_path, "w") as file:
    file.write("NIQE: %f\n" % niqe)
    file.write("TIME: %f\n" % _time)