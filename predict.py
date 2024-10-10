import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as sm

from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask, get_training_params
from post_processing import process_and_refine_prediction  # Import the post-processing function

VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in VALID_EXTENSIONS

def gather_image_files(input_path):
    """Gather all valid image files from a directory."""
    if os.path.isfile(input_path) and is_image_file(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        # Gather all valid image files within the directory
        return [os.path.join(input_path, f) for f in os.listdir(input_path) if is_image_file(f)]
    else:
        raise ValueError(f"Input path {input_path} is not a valid file or directory containing images.")


def predict_img(net,
                full_img,
                device,
                img_scale=None,
                imgW=None,
                imgH=None,
                out_threshold=0.5):
    net.eval()
    img_info = f"Images scaling:  {img_scale}" if img_scale else f"Image dimensions: Width={imgW}, Height={imgH}"
    logging.info({img_info})
    img = torch.from_numpy(BasicDataset.preprocess(mask_values = None,
                                                   pil_img = full_img,
                                                   is_mask=False,
                                                   scale = img_scale if img_scale else None,
                                                   newW = imgW if imgW else None,
                                                   newH = imgH if imgH else None))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images or folder')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filename of input image or directory', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Directory to save output masks', required=True)
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    # scale and img size options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scale', '-s', type=float, help='Downscaling factor of the images')
    group.add_argument('--size', '-sz', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Width and Height of the images')

    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Gather all input files
    in_files = gather_image_files(args.input)

    # Ensure output is a directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # none resnet34 version:
    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net = sm.Unet('resnet50', 
                  encoder_weights='imagenet', 
                  classes=args.classes)
    net.n_channels = 3
    net.n_classes = args.classes
    net.bilinear = args.bilinear

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    predict_params = get_training_params(args)

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           **predict_params,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        post_processed_mask = process_and_refine_prediction(mask,
                                                    threshold=args.mask_threshold,
                                                    min_size=500,  # Modify based on your application
                                                    kernel_size=3)

        if not args.no_save:
            out_filename = output_dir / f'{Path(filename).stem}_OUT.png'
            result = mask_to_image(post_processed_mask, mask_values)  # Save post-processed mask
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, post_processed_mask)  # Visualize post-processed mask
