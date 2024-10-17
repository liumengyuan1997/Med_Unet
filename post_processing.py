import numpy as np
from scipy.ndimage import label, binary_fill_holes, binary_dilation, binary_erosion
from skimage.morphology import disk

def post_process_prediction(mask_pred, threshold=0.5, min_size=3):
    """
    Post-process the predicted mask.
    Apply thresholding, remove small objects, and return the refined mask.
    """
    # Apply threshold to prediction
    mask = mask_pred > threshold  # Reduced threshold to capture more true positives
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
        if np.sum(label_mask) >= min_size:  # Reduced min_size to capture smaller objects
            processed_mask[label_mask] = 1

    return processed_mask

def process_and_refine_prediction(mask, threshold=0.5, min_size=3, dilation_iterations=2, erosion_iterations=1, kernel_size=2):
    """
    Refine the mask using thresholding, size filtering, morphological operations (dilation, erosion).
    """
    # Step 1: Post-process the mask by removing small objects
    processed_mask = post_process_prediction(mask, threshold=threshold, min_size=min_size)

    # Step 2: Apply morphological operation to fill holes in the mask
    # Fill small enclosed holes within the mask
    filled_mask = binary_fill_holes(processed_mask).astype(np.uint8)

    # Step 3: Perform morphological dilation followed by erosion
    # Create a structuring element (disk-shaped kernel) for dilation and erosion
    struct_elem = disk(kernel_size)

    # Apply dilation to the mask to fill small gaps, multiple iterations for better coverage
    dilated_mask = filled_mask
    for _ in range(dilation_iterations):
        dilated_mask = binary_dilation(dilated_mask, structure=struct_elem)

    # Apply erosion to the mask to remove small noise after dilation, fewer iterations for finer control
    eroded_mask = dilated_mask
    for _ in range(erosion_iterations):
        eroded_mask = binary_erosion(eroded_mask, structure=struct_elem)

    return eroded_mask
