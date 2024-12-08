import numpy as np
import os
from PIL import Image
from skimage import measure
from stl import mesh
import re
import json

def png_to_stl_simple(png_folder, stl_filename, last_processed_file='last_processed.json'):
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
        if file_index > last_index:
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

    # Use marching_cubes to create a 3D mesh from the 3D volume
    print("Generating 3D mesh using marching cubes...")
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)
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

# Example usage:
png_folder = '/home/keith/Downloads/NU Works/Research/Data/Poster/09S1 old'
stl_filename = 'output_modelonly2.stl'
png_to_stl_simple(png_folder, stl_filename)
