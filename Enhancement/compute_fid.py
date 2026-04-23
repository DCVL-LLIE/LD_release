"""
pip install pytorch-fid  

Usage
-----
from compute_fid import compute_fid
fid_val = compute_fid(real_dir, fake_dir, device='cuda:0')
print(f"FID = {fid_val:.4f}")
"""
import os, pathlib
import numpy as np
import torch
from PIL import Image
from scipy import linalg
import torchvision.transforms as TF
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
from typing import Sequence, Tuple, Union

IMAGE_EXT = {"bmp","jpg","jpeg","pgm","png","ppm","tif","tiff","webp"}

# ────────────────────────────────────────────────────────────
# 내부: 이미지 → Inception 활성값
# ────────────────────────────────────────────────────────────
class _ImageFolder(torch.utils.data.Dataset):
    def __init__(self, files, tfm):
        self.files, self.tfm = files, tfm
    def __len__(self):        return len(self.files)
    def __getitem__(self, i): return self.tfm(Image.open(self.files[i]).convert("RGB"))

def _activations(files:Sequence[pathlib.Path],
                 model:torch.nn.Module,
                 device:str,
                 batch:int=64,
                 dims:int=2048,
                 workers:int=4)->np.ndarray:
    if not files: raise ValueError("No images found for FID evaluation.")
    ds = _ImageFolder(files, TF.ToTensor())
    dl = torch.utils.data.DataLoader(ds, batch_size=min(batch,len(files)),
                                     shuffle=False, num_workers=workers, pin_memory=True)
    preds = np.empty((len(files), dims))
    model.eval()
    start = 0
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            feat = model(x)[0]
            if feat.shape[-1] != 1 or feat.shape[-2] != 1:
                feat = adaptive_avg_pool2d(feat, (1,1))
            feat = feat.squeeze(-1).squeeze(-1).cpu().numpy()
            preds[start:start+feat.shape[0]] = feat
            start += feat.shape[0]
    return preds

def _stats(path:Union[str,pathlib.Path],
           model, device, dims, batch, workers)->Tuple[np.ndarray,np.ndarray]:
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        with np.load(path) as f: mu, sigma = f["mu"], f["sigma"]
        return mu, sigma

    imgs = sorted([p for ext in IMAGE_EXT for p in path.glob(f"*.{ext}")])
    acts = _activations(imgs, model, device, batch, dims, workers)
    return acts.mean(0), np.cov(acts, rowvar=False)

def _frechet(mu1,s1,mu2,s2,eps=1e-6)->float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0])*eps
        covmean = linalg.sqrtm((s1+offset) @ (s2+offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff@diff + np.trace(s1 + s2 - 2*covmean)

# ────────────────────────────────────────────────────────────
# 공개 함수: compute_fid
# ────────────────────────────────────────────────────────────
def compute_fid(
        real_path:str,
        fake_path:str,
        device:str = None,
        dims:int   = 2048,
        batch:int  = 64,
        workers:int= 4) -> float:

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    block = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block]).to(device)

    mu1,s1 = _stats(real_path, model, device, dims, batch, workers)
    mu2,s2 = _stats(fake_path, model, device, dims, batch, workers)
    return _frechet(mu1, s1, mu2, s2)
