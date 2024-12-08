import os
from PIL import Image

def convert_4channel_to_3channel(image_path):
    """
    Convert a 4-channel image (RGBA) to 3-channel (RGB) by dropping the alpha channel.
    """
    img = Image.open(image_path)
    
    if img.mode == 'RGBA':
        # Convert from RGBA to RGB (drop alpha channel)
        img = img.convert('RGB')
    return img

def process_images_in_folder(folder_path):
    """
    Process all images in the given folder, convert 4-channel images to 3-channel.
    Replace the original image with the converted one.
    """
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(image_path):
            try:
                # Convert and replace the original image
                img = convert_4channel_to_3channel(image_path)
                img.save(image_path)  # Overwrite the original image
                print(f"Replaced original image with 3-channel version: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# Example usage
folder_path = "/home/keith/Downloads/NU Works/Research/Data/Poster/unified_MRI_with_mask/TD01_S1/TD01_S1_MRI_axial_mask"  # Replace with your folder path
process_images_in_folder(folder_path)
