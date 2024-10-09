
import os
import torch
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from utils.utils import plot_img_and_mask
from post_processing import process_and_refine_prediction
from utils.data_loading import BasicDataset
from utils.boundary_loss import ABL
from utils.dice_score import dice_loss
from predict import predict_img

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

    criterion = ABL()  # Boundary loss for validation, if needed
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

            # Predict the mask using the predict_img function from predict.py
            mask_pred = predict_img(net=net, full_img=img, device=device, out_threshold=mask_threshold)

            # Apply post-processing if enabled
            if post_process:
                processed_mask = process_and_refine_prediction(mask_pred, threshold=mask_threshold)
            else:
                processed_mask = mask_pred

            # Save the predicted mask
            mask_save_path = os.path.join(output_folder, f"{Path(img_file).stem}_pred.png")
            result_img = Image.fromarray((processed_mask * 255).astype(np.uint8))
            result_img.save(mask_save_path)
            logging.info(f'Saved predicted mask to: {mask_save_path}')

            # Optional validation if ground truth is available
            if gt_folder:
                gt_path = os.path.join(gt_folder, img_file)
                if os.path.exists(gt_path):
                    gt_img = Image.open(gt_path)
                    gt_array = np.array(gt_img)
                    val_loss = criterion(processed_mask, gt_array)
                    val_dice = dice_loss(processed_mask, gt_array)
                    total_val_loss += val_loss.item()
                    total_val_dice += val_dice.item()
                    n_val_images += 1

    # Validation statistics
    if n_val_images > 0:
        avg_val_loss = total_val_loss / n_val_images
        avg_val_dice = total_val_dice / n_val_images
        logging.info(f"Average validation loss: {avg_val_loss}")
        logging.info(f"Average validation Dice score: {avg_val_dice}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Predict masks for images in a folder')
    parser.add_argument('--model', type=str, help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, help='Path to the input folder containing images')
    parser.add_argument('--output', type=str, help='Path to the output folder for predicted masks')
    parser.add_argument('--gt', type=str, default=None, help='Optional path to ground truth masks for validation')
    parser.add_argument('--mask-threshold', type=float, default=0.5, help='Threshold for mask binarization')
    parser.add_argument('--no-post-process', action='store_true', help='Disable post-processing of the predicted masks')
    args = parser.parse_args()

    # Load the model
    logging.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = torch.load(args.model, map_location=device)
    net.to(device)

    # Call the prediction function
    predict_all_images(
        net=net,
        input_folder=args.input,
        output_folder=args.output,
        device=device,
        mask_threshold=args.mask_threshold,
        post_process=not args.no_post_process,
        gt_folder=args.gt
    )
