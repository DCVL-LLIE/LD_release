import os
from PIL import Image
import numpy as np
from basicsr.metrics.niqe import calculate_niqe
import argparse

def calculate_niqe_for_folder(image_folder, crop_border=0, input_order='HWC', convert_to='y', save_results=True):
    """
    Calculate NIQE scores for all images in a folder and their average.

    Args:
        image_folder (str): Path to the folder containing images.
        crop_border (int): Border size to crop before NIQE calculation.
        input_order (str): Input order of the image ('HW', 'HWC', 'CHW').
        convert_to (str): Color conversion type ('y' for YCbCr Y channel or 'gray').
        save_results (bool): Whether to save the NIQE scores to a file.

    Returns:
        dict: A dictionary containing image names and their corresponding NIQE scores.
        float: The average NIQE score.
    """
    niqe_scores = {}
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    if not image_files:
        print("No valid image files found in the specified folder.")
        return niqe_scores, None

    # Process each image
    for image_path in image_files:
        try:
            # Load image and convert to numpy array
            img = np.array(Image.open(image_path).convert('RGB'))  # Ensure RGB format
            
            # Ensure the image range is [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.max() > 255:
                img = img.clip(0, 255).astype(np.uint8)

            # Calculate NIQE score
            niqe_score = calculate_niqe(img, crop_border, input_order=input_order, convert_to=convert_to)

            # Ensure NIQE score is a float (handle potential ndarray output)
            niqe_score = float(np.squeeze(niqe_score))
            
            niqe_scores[os.path.basename(image_path)] = niqe_score
            print(f"NIQE for {image_path}: {niqe_score:.4f}")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Calculate the average NIQE score
    if niqe_scores:
        avg_niqe = sum(niqe_scores.values()) / len(niqe_scores)
        print(f"Average NIQE score: {avg_niqe:.4f}")
    else:
        avg_niqe = None

    # Optionally save results to a text file
    if save_results:
        results_file = os.path.join(image_folder, "niqe_scores.txt")
        with open(results_file, 'w') as f:
            for img_name, score in niqe_scores.items():
                f.write(f"{img_name}: {score:.4f}\n")
            if avg_niqe is not None:
                f.write(f"\nAverage NIQE: {avg_niqe:.4f}\n")
        print(f"NIQE scores saved to {results_file}")
    
    return niqe_scores, avg_niqe

if __name__ == '__main__':
    image_folder = "/mnt/JJH/ICCV2025/results/unpaired/LIME_"
    niqe_scores, avg_niqe = calculate_niqe_for_folder(image_folder, crop_border=0, input_order='HWC', convert_to='y')

    if avg_niqe is not None:
        print(f"Average NIQE score: {avg_niqe:.4f}")