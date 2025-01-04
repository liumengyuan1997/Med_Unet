import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import segmentation_models_pytorch as sm

def load_models(model_paths, device, num_classes):
    """
    Load models from specified paths.
    """
    models = []
    for path in model_paths:
        model = sm.Unet('resnet50', encoder_weights=None, classes=num_classes)
        state_dict = torch.load(path, map_location=device)
        state_dict.pop('mask_values', None)  # Ignore extra keys if any
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def preprocess_image(image_path, img_size, device):
    """
    Preprocess the image for prediction.
    """
    img = Image.open(image_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    img = transform(img).unsqueeze(0)
    return img.to(device), original_size

def predict_with_models(models, image_tensor, original_size):
    """
    Predict the mask using multiple models and resize to original dimensions.
    """
    predictions = []
    target_size = (original_size[1], original_size[0])
    
    for model in models:
        with torch.no_grad():
            output = model(image_tensor).cpu()
            mask = torch.sigmoid(output)[:, 1].squeeze()
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                 size=target_size, 
                                 mode='bilinear', 
                                 align_corners=False).squeeze()
            predictions.append((mask > 0.5).numpy())
    
    predictions = np.stack(predictions)
    majority_vote_mask = (np.sum(predictions, axis=0) >= 3).astype(np.uint8)
    return majority_vote_mask

def save_mask(mask, output_path):
    """
    Save the predicted mask as an image.
    """
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    
    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask.shape}")
    
    mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_img.save(output_path)

def process_folder(input_folder, output_folder, models, img_size, device):
    """
    Process all images in a folder and apply majority voting.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of valid image files
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        
        try:
            # Preprocess image
            image_tensor, original_size = preprocess_image(image_path, img_size, device)
            
            # Predict using majority voting
            majority_vote_mask = predict_with_models(models, image_tensor, original_size)
            
            # Save the majority-voted mask
            save_mask(majority_vote_mask, output_path)
            print(f"Processed and saved: {output_path}")
            
            # Clear intermediate tensors to free memory
            del image_tensor, majority_vote_mask
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    # Define paths to your models
    model_paths = [
        './checkpoints/record/checkpoint_epoch7401_9620.pth',
        './checkpoints/record/checkpoint_epoch8702_9731.pth',
        './checkpoints/record/checkpoint_epoch9003_9754.pth',
        './checkpoints/record/checkpoint_epoch6004_9745.pth',
        './checkpoints/record/checkpoint_epoch9405_9640.pth'
    ]
    
    # Define input folder and output folder
    input_folder = './test/09S1_images'
    output_folder = './test/09S1__output'
    
    # Define parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    img_size = (256, 256)
    
    # Load models
    models = load_models(model_paths, device, num_classes)
    
    # Process all images in the folder
    process_folder(input_folder, output_folder, models, img_size, device)
    
    print(f"All images processed. Results saved to {output_folder}")
