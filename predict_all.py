import os
import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
import numpy as np
import segmentation_models_pytorch as sm
import torch.nn.functional as F

from utils.utils import plot_img_and_mask
from post_processing import process_and_refine_prediction
from utils.data_loading import BasicDataset
from utils.boundary_loss import ABL
from utils.dice_score import dice_loss

def predict_all_images(net, input_folder, output_folder, device, mask_threshold=0.5, post_process=True, gt_folder=None):
    """
    Predict masks for all images in a folder, and optionally calculate validation scores if ground truth is provided.
    
    Parameters:
    - net: The trained segmentation model.
    - input_folder: The path to the folder containing input images.
    - output_folder: The path to the folder where predicted masks will be saved.
    - device: The device to run the model on.
    - mask_threshold: The threshold for binarizing masks.
    - post_process: Whether to apply post-processing to the predicted masks.
    - gt_folder: Optional path to a folder containing ground truth masks for validation.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    criterion = ABL()
    total_val_loss = 0
    total_val_dice = 0
    n_val_images = 0

    for img_file in os.listdir(input_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, img_file)
            logging.info(f'Predicting image: {img_path}')

            # Load input image
            img = Image.open(img_path)
            img_array = np.array(img)

            # Predict the mask
            mask_pred = predict_img(net=net, full_img=img, device=device, out_threshold=mask_threshold)

            # Apply post-processing if enabled
            if post_process:
                processed_mask = process_and_refine_prediction(mask_pred, threshold=mask_threshold)
            else:
                processed_mask = mask_pred

            # Save the predicted mask
            output_path = os.path.join(output_folder, f"{Path(img_file).stem}_mask.png")
            result = Image.fromarray(processed_mask.astype(np.uint8) * 255)  # Convert binary mask to image
            result.save(output_path)
            logging.info(f'Saved mask to {output_path}')

            # If ground truth is provided, calculate validation scores
            if gt_folder:
                gt_path = os.path.join(gt_folder, img_file)
                if os.path.exists(gt_path):
                    true_mask = np.array(Image.open(gt_path))
                    true_mask_tensor = torch.from_numpy(true_mask).to(device=device, dtype=torch.long)

                    # Calculate validation loss and dice score
                    processed_mask_tensor = torch.from_numpy(processed_mask).to(device=device, dtype=torch.float32)

                    if net.n_classes > 1:
                        val_loss = 0.1 * criterion(processed_mask_tensor.unsqueeze(0), true_mask_tensor.unsqueeze(0))
                        val_loss += 0.9 * dice_loss(
                            F.softmax(processed_mask_tensor.unsqueeze(0), dim=1),
                            F.one_hot(true_mask_tensor.unsqueeze(0), net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    else:
                        val_loss = criterion(processed_mask_tensor.unsqueeze(0), true_mask_tensor.unsqueeze(0))
                        val_loss += dice_loss(
                            torch.sigmoid(processed_mask_tensor.unsqueeze(0)),
                            true_mask_tensor.float().unsqueeze(0),
                            multiclass=False
                        )

                    total_val_loss += val_loss.item()
                    total_val_dice += (1 - val_loss.item())  # Approximate dice score
                    n_val_images += 1

    # If ground truth was provided, print the average validation scores
    if gt_folder and n_val_images > 0:
        avg_val_loss = total_val_loss / n_val_images
        avg_val_dice = total_val_dice / n_val_images
        logging.info(f'Average Validation Loss: {avg_val_loss}')
        logging.info(f'Average Dice Score: {avg_val_dice}')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks for all images in a folder')
    parser.add_argument('--input-folder', '-i', type=str, required=True, help='Path to the input folder with images')
    parser.add_argument('--output-folder', '-o', type=str, required=True, help='Path to the output folder for saving masks')
    parser.add_argument('--model', '-m', default='MODEL.pth', type=str, help='Path to the model .pth file')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Threshold for mask binarization')
    parser.add_argument('--no-post-process', '-n', action='store_true', help='Disable post-processing of masks')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of output classes for segmentation')
    parser.add_argument('--gt-folder', '-g', type=str, help='Path to ground truth masks for validation')
    
    return parser.parse_args()

def predict_img(net, full_img, device, out_threshold=0.5):
    net.eval()

    # Preprocess the image
    img = torch.from_numpy(BasicDataset.preprocess(mask_values=None, pil_img=full_img, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = torch.nn.functional.interpolate(output, size=(full_img.size[1], full_img.size[0]), mode='bilinear')
        
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load the model
    net = sm.Unet('resnet50', encoder_weights='imagenet', classes=args.classes)
    net.n_channels = 3
    net.n_classes = args.classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model from {args.model}')
    net.to(device=device)
    
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded successfully')

    # Predict all images in the input folder, and optionally validate against ground truth
    predict_all_images(
        net=net,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        device=device,
        mask_threshold=args.mask_threshold,
        post_process=not args.no_post_process,
        gt_folder=args.gt_folder  # Ground truth folder for validation
    )
