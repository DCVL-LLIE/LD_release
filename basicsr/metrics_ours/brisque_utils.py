# brisque_utility.py (v2 - Corrected URL)

import cv2
import numpy as np
import scipy.stats
import scipy.special
import scipy.optimize
from os import path
import urllib.request

# --- Model files and parameters ---
# THIS URL HAS BEEN UPDATED TO A WORKING ONE.
_MODEL_URL = 'https://github.com/jeong-tae/brisque/raw/master/models/brisque_model_live.yml'
_MODEL_PATH = path.join(path.dirname(path.abspath(__file__)), 'brisque_model_live.yml')

def _download_model_if_needed():
    """Downloads the BRISQUE model file from a URL if it doesn't exist."""
    if not path.exists(_MODEL_PATH):
        print(f"BRISQUE model not found. Downloading from new URL...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise IOError("Could not download the BRISQUE model file.")

def get_brisque_score(image_bgr_uint8):
    """
    Calculates the BRISQUE score for a given BGR image.
    Image should be a numpy array in BGR uint8 format.
    """
    # 1. Ensure the model file is available
    _download_model_if_needed()

    # 2. Convert to grayscale
    image_gray = cv2.cvtColor(image_bgr_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 3. Compute features
    features = _compute_features(image_gray)

    # 4. Load model and predict score
    try:
        svm = cv2.ml.SVM_load(_MODEL_PATH)
        # The result is a tuple, e.g., (13.24, array([[13.24...]], dtype=float32))
        # We only need the first element.
        score = svm.predict(features)[0]
        return float(score)
    except cv2.error as e:
        # Provide a more helpful error message if the model file is corrupt
        if "bad argument" in str(e).lower():
             raise IOError(f"The BRISQUE model file at '{_MODEL_PATH}' seems to be corrupt or invalid. Please delete it and run the script again to re-download.")
        else:
             raise e


def _mscn_filter(image, kernel_size=7, sigma=7.0/6.0):
    """Calculates the MSCN coefficients."""
    C = 1.0
    # Gaussian blur
    g_blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    g_blur_sq = g_blur * g_blur
    # mu
    mu = g_blur
    # sigma
    sigma = np.sqrt(abs(cv2.GaussianBlur(image*image, (kernel_size, kernel_size), sigma) - g_blur_sq))
    
    mscn = (image - mu) / (sigma + C)
    return mscn


def _compute_features(image):
    """Computes the 36-dimensional feature vector for BRISQUE."""
    # Scale 1 (original)
    mscn1 = _mscn_filter(image)
    alpha, overall_std, left_std, right_std = _ggd_params(mscn1)
    feats1 = [alpha, overall_std]

    # Paired product features
    paired_products = _paired_products(mscn1)
    alpha_p, left_std_p, right_std_p = _aggd_params(paired_products)
    feats2 = [alpha_p, left_std_p, right_std_p]

    # Scale 2 (downscaled by 2)
    image_scaled = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    mscn2 = _mscn_filter(image_scaled)
    alpha2, overall_std2, left_std2, right_std2 = _ggd_params(mscn2)
    feats3 = [alpha2, overall_std2]

    # Paired product features for scale 2
    paired_products2 = _paired_products(mscn2)
    alpha_p2, left_std_p2, right_std_p2 = _aggd_params(paired_products2)
    feats4 = [alpha_p2, left_std_p2, right_std_p2]

    return np.array([*feats1, *feats2, *feats3, *feats4], dtype=np.float32).reshape(1, -1)


def _ggd_params(im):
    gamma_range = np.arange(0.2, 10.0, 0.001)
    # Estimate GGD parameters
    e_ggd = lambda x, g, s: (1.0/g) * (np.sum((np.abs(x)/s)**g) / len(x)) - 1
    solution = scipy.optimize.fsolve(lambda g: e_ggd(im, g, im.std()), 1)[0]
    gamma = solution

    sigma_sq = np.mean(im**2)
    sigma = np.sqrt(sigma_sq)
    
    # Asymmetric GGD
    pos_mask = im > 0
    neg_mask = im < 0
    left_std = np.sqrt(np.mean(im[neg_mask]**2)) if np.any(neg_mask) else 0
    right_std = np.sqrt(np.mean(im[pos_mask]**2)) if np.any(pos_mask) else 0
    
    return gamma, sigma, left_std, right_std


def _aggd_params(im):
    # Estimate AGGD parameters
    e_aggd = lambda x, al, sigma_l, sigma_r: (sigma_r - sigma_l) * (scipy.special.gamma(2.0/x)/scipy.special.gamma(1.0/x)) * np.mean(im) + \
            (sigma_l * np.mean(im[im<0])) + (sigma_r * np.mean(im[im>0]))
    
    pos_mask = im > 0
    neg_mask = im < 0
    
    left_std = np.sqrt(np.mean(im[neg_mask]**2)) if np.any(neg_mask) else 0
    right_std = np.sqrt(np.mean(im[pos_mask]**2)) if np.any(pos_mask) else 0
    mean = np.mean(im)

    if left_std == 0 or right_std == 0 or mean == 0:
        return 0, 0, 0

    solution = scipy.optimize.fsolve(lambda al: e_ggd(al, al, left_std, right_std), 1, maxfev=1000)[0]
    alpha = solution

    return alpha, left_std, right_std


def _paired_products(im):
    # Paired product features
    # H (horizontal)
    shift_right = np.roll(im, 1, axis=1)
    shift_right[:, 0] = im[:, 0]
    pp_h = im * shift_right
    # V (vertical)
    shift_down = np.roll(im, 1, axis=0)
    shift_down[0, :] = im[0, :]
    pp_v = im * shift_down
    # D1 (main-diagonal)
    shift_d1 = np.roll(np.roll(im, 1, axis=0), 1, axis=1)
    shift_d1[0, :] = im[0, :]
    shift_d1[:, 0] = im[:, 0]
    pp_d1 = im * shift_d1
    # D2 (anti-diagonal)
    shift_d2 = np.roll(np.roll(im, -1, axis=0), 1, axis=1)
    shift_d2[-1, :] = im[-1, :]
    shift_d2[:, 0] = im[:, 0]
    pp_d2 = im * shift_d2
    
    return np.concatenate([pp_h.ravel(), pp_v.ravel(), pp_d1.ravel(), pp_d2.ravel()])