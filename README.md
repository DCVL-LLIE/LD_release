# Low-light Image Enhancement via Distribution of Latent Transitions

This repository provides the official PyTorch implementation of the paper:  
**["Low-light Image Enhancement via Distribution of Latent Transitions"](#)** 

**Jaehun Jung** and [**Wonjun Kim** (Corresponding Author)](https://sites.google.com/view/dcvl)  
Journal of Visual Communication and Image Representation (JVCIR)

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

## 📂 Datasets

All datasets used for **training and evaluation** can be downloaded from this [Repository](https://github.com/caiyuanhao1998/retinexformer).

Please organize the downloaded datasets in the `./data/` directory as shown below:

```text
./data/
├── LOLv1/
│   ├── Train/
│   │   ├── input/
│   │   └── target/
│   └── Test/
│       ├── input/
│       └── target/
├── LOLv2/
│   ├── Real_captured/
│   │   ├── Train/
│   │   │   ├── Low/
│   │   │   └── Normal/
│   │   └── Test/
│   │       ├── Low/
│   │       └── Normal/
│   └── Synthetic/
│       ├── Train/
│       │   ├── Low/
│       │   └── Normal/
│       └── Test/
│           ├── Low/
│           └── Normal/
└── SDSD/
    ├── indoor_static_np/
    │   ├── input/
    │   └── GT/
    └── outdoor_static_np/
        ├── input/
        └── GT/
```

## Results
### Qualitative results
![..](figure.svg)
