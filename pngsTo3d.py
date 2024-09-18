import numpy as np
import os
from skimage import measure
from PIL import Image
from stl import mesh
from scipy.ndimage import binary_fill_holes
import re

def png_to_stl(png_folder, stl_filename):
    # Load PNG files and stack them into a 3D volume
    images = []
    png_files = sorted([os.path.join(png_folder, f) for f in os.listdir(png_folder) if f.endswith('.png')],
                         key=lambda x: int(re.findall(r'\d+', x)[0]))
    for filename in png_files:
        if filename.endswith('.png'):
            img = Image.open(os.path.join(png_folder, filename)).convert('L')
            images.append(np.array(img))
    
    # Convert list of 2D images into a 3D numpy array (Z, Y, X)
    volume = np.stack(images, axis=0)
    print(f"Volume shape: {volume.shape}")

    # Apply binary fill holes to ensure there are no gaps in the volume
    volume_filled = binary_fill_holes(volume > 0)

    # Use marching_cubes to create a 3D mesh from the 3D volume
    verts, faces, normals, values = measure.marching_cubes(volume_filled, level=0.5)
    print(f"Number of vertices: {len(verts)}")
    print(f"Number of faces: {len(faces)}")

    # Create a mesh object for STL export
    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = verts[f[j], :]

    # Save the mesh to an STL file
    mesh_data.save(stl_filename)
    print(f"STL file saved as {stl_filename}")
  

# Example usage:
png_folder = '/outputs_axial_01s2'
stl_filename = '/output_model_axial_01s2.stl'
png_to_stl(png_folder, stl_filename)
