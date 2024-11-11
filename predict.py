import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as sm

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask, get_training_params
from archs import UKAN
import cv2

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
        # output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def restore_mask_to_original_size(cropped_mask,
                                  original_size,
                                  crop_size,
                                  img_scale=None,
                                  imgW=None,
                                  imgH=None):

    original_width = int(cropped_mask.shape[1] / img_scale)
    original_height = int(cropped_mask.shape[0] / img_scale)

    resized_mask = cv2.resize(cropped_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    orig_w, orig_h = original_size

    restored_mask = torch.from_numpy(resized_mask)
    center_crop = transforms.CenterCrop((orig_h, orig_w))
    restored_mask = center_crop(restored_mask.unsqueeze(0)).squeeze(0)

    return restored_mask.numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    # scale and img size options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scale', '-s', type=float, help='Downscaling factor of the images')
    group.add_argument('--size', '-sz', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'), help='Width and Height of the images')

    # old version
    # parser.add_argument('--scale', '-s', type=float, default=1.0,
    #                     help='Scale factor for the input images')
    # parser.add_argument('--imgW', '-iw', type=int, default=224)
    # parser.add_argument('--imgH', '-ih', type=int, default=224)
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


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

    in_files = args.input
    out_files = get_output_filenames(args)

    # none resnet34 version:
    # net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    
    # net = sm.Unet('resnet50', 
    #               encoder_weights='imagenet', 
    #               classes=args.classes)
    # net.n_channels = 3
    # net.n_classes = args.classes
    # net.bilinear = args.bilinear

    net = UKAN(num_classes=args.classes)

    net = net.to(memory_format=torch.channels_last)

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
        
        mask = restore_mask_to_original_size(mask, img.size, 224, **predict_params)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
