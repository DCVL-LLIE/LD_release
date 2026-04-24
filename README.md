# Low-light Image Enhancement via Distribution of Latent Transitions

This repository provides the official PyTorch implementation of the paper:  
**["Low-light Image Enhancement via Distribution of Latent Transitions"](#)** 

**Jaehun Jung** and [**Wonjun Kim** (Corresponding Author)](https://sites.google.com/view/dcvl)  
Journal of Visual Communication and Image Representation (JVCIR)

## Results
### Qualitative results
![..](figure.svg)

## ⚙️ Installation

### 🐍 Environment Setup
```bash
conda create -n Retinexformer python=3.7
conda activate Retinexformer
```

### 📦 Install Dependencies
```bash
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

### 🛠️ Build BasicSR
```bash
python setup.py develop --no_cuda_ext
```
