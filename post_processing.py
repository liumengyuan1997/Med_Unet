import numpy as np
import torch
import cv2
from scipy.ndimage import label

def post_process_prediction(mask_pred, threshold=0.5, min_size=500):
    """
    Post-process the predicted mask to remove false positives.

    Parameters:
        mask_pred (torch.Tensor or np.ndarray): Predicted mask, shape (H, W) or (C, H, W).
        threshold (float): Confidence threshold for binarizing the mask.
        min_size (int): Minimum size of connected components to retain.

    Returns:
        np.ndarray: The post-processed binary mask.
    """
    # Ensure mask is in numpy format if it's a tensor
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = mask_pred.detach().cpu().numpy()

    # If the mask has multiple channels, we assume the prediction is a probability map.
    if len(mask_pred.shape) == 3 and mask_pred.shape[0] > 1:
        mask_pred = np.argmax(mask_pred, axis=0)  # Convert to single-channel mask
    elif len(mask_pred.shape) == 3:
        mask_pred = mask_pred[0]  # Binary prediction for single-channel models

    # Thresholding to create a binary mask (binarize the prediction)
    binary_mask = (mask_pred > threshold).astype(np.uint8)

    # Perform connected component analysis
    num_labels, labels = label(binary_mask)
    
    # Remove small objects (false positives)
    for label_idx in range(1, num_labels + 1):
        component = labels == label_idx
        if np.sum(component) < min_size:
            binary_mask[component] = 0  # Remove small components

    return binary_mask

def morphological_post_process(mask, kernel_size=3, operation='open'):
    """
    Apply morphological operations to remove noise from the binary mask.
    
    Parameters:
        mask (np.ndarray): Binary mask, shape (H, W).
        kernel_size (int): Size of the structuring element for morphological operations.
        operation (str): Type of morphological operation ('open', 'close', 'dilate', 'erode').
    
    Returns:
        np.ndarray: Morphologically processed mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'open':
        # Opening removes small noise in the foreground
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        # Closing removes small holes in the foreground
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'dilate':
        processed_mask = cv2.dilate(mask, kernel)
    elif operation == 'erode':
        processed_mask = cv2.erode(mask, kernel)
    else:
        raise ValueError("Operation must be 'open', 'close', 'dilate', or 'erode'")
    
    return processed_mask

def process_and_refine_prediction(mask_pred, threshold=0.5, min_size=500, kernel_size=3):
    """
    Combined function that applies both post-processing techniques.
    
    Parameters:
        mask_pred (torch.Tensor or np.ndarray): Predicted mask, shape (H, W) or (C, H, W).
        threshold (float): Confidence threshold for binarizing the mask.
        min_size (int): Minimum size of connected components to retain.
        kernel_size (int): Size of the structuring element for morphological operations.
    
    Returns:
        np.ndarray: The post-processed and refined mask.
    """
    # Apply post-processing to remove false positives
    processed_mask = post_process_prediction(mask_pred, threshold=threshold, min_size=min_size)
    
    # Apply morphological operation (e.g., opening to remove noise)
    refined_mask = morphological_post_process(processed_mask, kernel_size=kernel_size, operation='open')
    
    return refined_mask