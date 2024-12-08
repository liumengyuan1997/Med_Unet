import os
import re
import shutil

def rename_and_copy_images(source_folder, target_folder):
    """
    Rename image files from the source folder and save them to the target folder.
    New filenames follow the format TD01_S1_MRI_axial_origin_{last_three_digits}.png.
    
    Args:
        source_folder (str): Path to the folder containing original image files.
        target_folder (str): Path to the folder where renamed files will be saved.
    """
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)
    
    for filename in os.listdir(source_folder):
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue  # Skip non-image files
        
        # Match the pattern to extract the necessary parts of the filename
        match = re.match(r"(TD\d+_S\d+)_L_.*?_(\d{3})", filename)  # Ensure we capture exactly 3 digits
        if match:
            identifier_1, last_three_digits = match.groups()
            
            # Debugging: Print what was captured
            print(f"Matched: identifier_1={identifier_1}, last_three_digits={last_three_digits}")
            
            # Construct the new filename
            new_name = f"{identifier_1}_MRI_axial_origin_{last_three_digits}{os.path.splitext(filename)[1]}"
            
            # Build full source and destination paths
            src = os.path.join(source_folder, filename)
            dst = os.path.join(target_folder, new_name)
            
            # Avoid overwriting existing files
            if os.path.exists(dst):
                print(f"Warning: {new_name} already exists. Skipping...")
                continue
            
            # Copy the file to the new location with the new name
            shutil.copy(src, dst)
            print(f"Copied and renamed: {filename} -> {new_name}")
        else:
            print(f"Skipped: {filename} (Pattern not matched)")

# Example usage:
source_folder = "/home/keith/Downloads/NU Works/Research/Data/Poster/unified_MRI_with_mask/TD01_S1/TD01_S1_MRI_axial_mask (copy)"
target_folder = "/home/keith/Downloads/NU Works/Research/Data/Poster/unified_MRI_with_mask/TD01_S1/TD01_S1_MRI_axial_mask"
rename_and_copy_images(source_folder, target_folder)
