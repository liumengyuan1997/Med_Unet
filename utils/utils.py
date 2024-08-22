import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.ndimage as ndimage


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

def get_training_params(args):
    if args.scale:
        return {
            'img_scale': args.scale
        }
    elif args.size:
        return {
            'imgW': args.size[0],
            'imgH': args.size[1]
        }
    else:
        # Default image dimensions
        return {
            'imgW': 224,
            'imgH': 224
        }

def compute_distance_map(masks):
    """
    Compute the distance map for each mask in the batch for both channels.
    
    Args:
        masks (torch.Tensor): A tensor of shape (batch_size, 2, x, y) containing binary masks.
    
    Returns:
        torch.Tensor: A tensor of shape (batch_size, 2, x, y) containing the distance maps.
    """
    batch_size = masks.size(0)
    num_channels = masks.size(1)
    distance_maps = []

    # Move masks to CPU and convert to numpy for processing
    masks_np = masks.detach().cpu().numpy()

    for i in range(batch_size):
        for j in range(num_channels):
            mask = masks_np[i, j]  # Extract single mask for each channel
            # Convert to binary mask (0 and 1 values)
            mask_binary = mask > 0.5

            # Compute the Euclidean distance transform
            distance_map = ndimage.distance_transform_edt(mask_binary)
            
            # Append the distance map for the current mask and channel
            distance_maps.append(distance_map)

    # Stack the distance maps and reshape to match the input shape
    distance_maps = np.stack(distance_maps).reshape(batch_size, num_channels, *masks.shape[2:])
    distance_maps_tensor = torch.tensor(distance_maps, dtype=torch.float32).to(masks.device)

    return distance_maps_tensor

def generateLossPlot(epochs, train_loss_ls, val_loss_ls):
    epochs_ls = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs_ls, train_loss_ls, 'b-', label='Training Loss')
    plt.plot(epochs_ls, val_loss_ls, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
