import os
import torch
import numpy as np
import cv2
import re
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Calculate the Dice coefficient between two binary masks, supporting batch processing.
    Args:
    - input (Tensor): The predicted mask (binary).
    - target (Tensor): The ground truth mask (binary).
    - reduce_batch_first (bool): Whether to calculate the Dice coefficient over the batch dimension first.
    - epsilon (float): Small value to avoid division by zero.
    
    Returns:
    - dice (float): The mean Dice score.
    """
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = (input).sum(dim=sum_dim) + (target).sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def load_image(image_path):
    """
    Load an image, binarize it, and convert it to a tensor (0 or 1).
    Args:
    - image_path (str): Path to the image.
    
    Returns:
    - mask (Tensor): The binary mask (0 and 1).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    _, binarized_mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)  # Convert to binary (0 or 1)
    return torch.tensor(binarized_mask, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

def match_files(pred_files, gt_files):
    """
    Match ground truth and predicted images based on pattern in their filenames.
    Args:
    - pred_files (list): List of predicted mask filenames.
    - gt_files (list): List of ground truth mask filenames.
    
    Returns:
    - matched_files (list of tuples): List of tuples where each tuple contains 
      the predicted file and the corresponding ground truth file.
    """
    matched_files = []
    pattern = re.compile(r"(\d{2}_S\d+).*?(\d{3})")
    
    for pred_file in pred_files:
        match_pred = pattern.search(pred_file)
        if match_pred:
            pred_id = match_pred.group(1) + match_pred.group(2)
            
            for gt_file in gt_files:
                match_gt = pattern.search(gt_file)
                if match_gt:
                    gt_id = match_gt.group(1) + match_gt.group(2)
                    if pred_id == gt_id:
                        matched_files.append((pred_file, gt_file))
                        break
    return matched_files

def evaluate_dice_score(pred_mask_path, gt_mask_path):
    """
    Calculate the average Dice score for a set of predicted and ground truth masks.
    Args:
    - pred_mask_path (str): Path to the folder containing predicted mask images.
    - gt_mask_path (str): Path to the folder containing ground truth mask images.
    
    Returns:
    - avg_dice (float): The average Dice score for all matched images.
    """
    pred_images = sorted(os.listdir(pred_mask_path))
    gt_images = sorted(os.listdir(gt_mask_path))
    
    matched_files = match_files(pred_images, gt_images)
    
    if not matched_files:
        print("No matching files found!")
        return 0.0

    dice_scores = []
    
    for pred_img_name, gt_img_name in matched_files:
        pred_mask = load_image(os.path.join(pred_mask_path, pred_img_name))
        gt_mask = load_image(os.path.join(gt_mask_path, gt_img_name))
        
        # Calculate Dice coefficient using dice_coeff function
        dice = dice_coeff(pred_mask, gt_mask)
        dice_scores.append(dice.item())  # Convert tensor to scalar value
    
    avg_dice = np.mean(dice_scores)
    return avg_dice

if __name__ == "__main__":
    # Example paths (replace with your actual directories)
    pred_mask_path = '/home/keith/Downloads/NU Works/Research/Data/Poster/09S1 old'
    gt_mask_path = '/home/keith/Downloads/NU Works/Research/Data/Poster/unified_MRI_with_mask/TD09_S1/TD09_S1_MRI_axial_mask'

    avg_dice = evaluate_dice_score(pred_mask_path, gt_mask_path)
    print(f"Average Dice Score: {avg_dice:.4f}")
