import numpy as np
from scipy.ndimage import label

def post_process_prediction(mask_pred, threshold=0.5, min_size=500):
    """
    Post-process the predicted mask.
    Apply thresholding, remove small objects, and return the refined mask.
    """
    # Apply threshold to prediction
    mask = mask_pred > threshold
    mask = mask.astype(np.uint8)  # Ensure mask is an integer array

    # Label connected components
    labeled_mask, num_labels = label(mask)

    # Ensure num_labels is an integer
    num_labels = int(num_labels)

    # Initialize processed mask
    processed_mask = np.zeros_like(mask, dtype=np.uint8)

    # Iterate over each label
    for label_idx in range(1, num_labels + 1):
        label_mask = (labeled_mask == label_idx)

        # Remove small objects based on size
        if np.sum(label_mask) >= min_size:
            processed_mask[label_mask] = 1

    return processed_mask


def process_and_refine_prediction(mask, threshold=0.5, min_size=500, kernel_size=3):
    """
    Refine the mask using thresholding, size filtering, and morphological operations.
    """
    processed_mask = post_process_prediction(mask, threshold=threshold, min_size=min_size)

    # Optionally apply morphological operations or other refinements here
    # ...

    return processed_mask
