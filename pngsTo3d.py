import numpy as np
import os
from skimage import measure, morphology
from PIL import Image
from stl import mesh
from scipy.ndimage import binary_fill_holes
import re
import json
from skimage.restoration import denoise_bilateral  # For Bilateral Smoothing

def png_to_stl(png_folder, stl_filename, last_processed_file='last_processed.json', smoothing_method=None):
    # Load or initialize last processed image index
    print("Initializing last processed index...")
    last_index = -1
    if os.path.exists(last_processed_file):
        with open(last_processed_file, 'r') as f:
            last_processed_data = json.load(f)
            last_index = last_processed_data.get('last_index', -1)

    # Load PNG files and filter based on last processed index
    print("Loading PNG files...")
    valid_png_files = []
    for f in os.listdir(png_folder):
        if f.endswith('.png') and re.search(r'_origin_(\d+)_OUT', f):
            valid_png_files.append(f)
        else:
            print(f"Skipping invalid filename: {f}")

    png_files = sorted(
        [os.path.join(png_folder, f) for f in valid_png_files],
        key=lambda x: int(re.findall(r'_origin_(\d+)_OUT', os.path.basename(x))[0])
    )

    images = []
    new_last_index = last_index
    for i, filename in enumerate(png_files):
        file_index = int(re.findall(r'_origin_(\d+)_OUT', os.path.basename(filename))[0])
        # if file_index > last_index:
        img = Image.open(filename).convert('L')
        images.append(np.array(img))
        new_last_index = max(new_last_index, file_index)
        if i % 50 == 0:
            print(f"Loaded {i+1}/{len(png_files)} images...")

    if not images:
        print("No new images to process.")
        return

    # Convert list of 2D images into a 3D numpy array (Z, Y, X)
    volume = np.stack(images, axis=0)
    print(f"Volume shape: {volume.shape}")

    # Apply binary fill holes to ensure there are no gaps in the volume
    print("Applying binary fill holes...")
    volume_filled = binary_fill_holes(volume > 0)

    # Post-processing: remove small objects (false positives)
    print("Removing small objects...")
    volume_cleaned = morphology.remove_small_objects(volume_filled, min_size=150)

    # Apply dilation followed by erosion to smooth the segmented regions
    print("Applying dilation and erosion...")
    volume_dilated = morphology.binary_dilation(volume_cleaned, morphology.ball(1))
    volume_processed = morphology.binary_erosion(volume_dilated, morphology.ball(1))

    # Optional Smoothing Steps
    if smoothing_method == 'curvature':
        print("Applying curvature flow smoothing...")
        volume_processed = curvature_smoothing(volume_processed)
    elif smoothing_method == 'bilateral':
        print("Applying bilateral smoothing (slice-by-slice)...")
        volume_processed = apply_bilateral_smoothing(volume_processed)
    elif smoothing_method == 'anisotropic':
        print("Applying anisotropic diffusion smoothing...")
        volume_processed = anisotropic_diffusion(volume_processed.astype(float), niter=5, kappa=50, gamma=0.1)

    # Use marching_cubes to create a 3D mesh from the 3D volume
    print("Generating 3D mesh using marching cubes...")
    verts, faces, normals, values = measure.marching_cubes(volume_processed, level=0.5)
    print(f"Number of vertices: {len(verts)}")
    print(f"Number of faces: {len(faces)}")

    # Create a mesh object for STL export
    print("Creating STL mesh data...")
    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = verts[f[j], :]
        if i % 1000 == 0:
            print(f"Processed {i+1}/{len(faces)} faces...")

    # Save the mesh to an STL file
    mesh_data.save(stl_filename)
    print(f"STL file saved as {stl_filename}")

    # Save the new last processed index
    with open(last_processed_file, 'w') as f:
        json.dump({'last_index': new_last_index}, f)
    print("Updated last processed index.")

# Curvature Flow Smoothing
def curvature_smoothing(volume, iterations=3):
    from skimage.filters import gaussian
    for iter_num in range(iterations):
        volume = gaussian(volume, sigma=1) * 0.9 + volume * 0.1  # Smoothing with Gaussian blend
        print(f"Curvature smoothing iteration {iter_num + 1}/{iterations} completed.")
    return volume

# Bilateral Smoothing applied slice-by-slice with reduced sigma values
def apply_bilateral_smoothing(volume):
    smoothed_volume = np.zeros_like(volume, dtype=float)
    for i in range(volume.shape[0]):
        smoothed_volume[i] = denoise_bilateral(volume[i].astype(float), sigma_color=0.02, sigma_spatial=7)
        if i % 50 == 0:
            print(f"Bilateral smoothing on slice {i+1}/{volume.shape[0]} completed.")
    return smoothed_volume

# Anisotropic Diffusion (basic implementation for demonstration)
def anisotropic_diffusion(volume, niter=5, kappa=50, gamma=0.1):
    def diffusion_step(vol, kappa, gamma):
        diff_x = np.diff(vol, axis=0)
        diff_y = np.diff(vol, axis=1)
        diff_z = np.diff(vol, axis=2)
        vol[:-1, :, :] += gamma * (diff_x / (1 + (diff_x/kappa)**2))
        vol[:, :-1, :] += gamma * (diff_y / (1 + (diff_y/kappa)**2))
        vol[:, :, :-1] += gamma * (diff_z / (1 + (diff_z/kappa)**2))
        return vol
    for iter_num in range(niter):
        volume = diffusion_step(volume, kappa, gamma)
        print(f"Anisotropic diffusion iteration {iter_num + 1}/{niter} completed.")
    return volume

# Example usage:
png_folder = '/home/zhangyanyu/unet/data/test09/predict'
stl_filename = '/home/zhangyanyu/unet/data/test09/old09s1_model.stl'
png_to_stl(png_folder, stl_filename, smoothing_method='curvature')  # Choose smoothing method: 'curvature', 'bilateral', or 'anisotropic'
# png_to_stl(png_folder, stl_filename)