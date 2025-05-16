# imports here for now during reorg
import h5py
import numpy as np
import trimesh
from skimage.measure import marching_cubes
from scipy.ndimage import binary_closing, gaussian_filter
from scipy.interpolate import interp1d

# TODO: implement TQDM logging
# from tqdm import tqdm

def neuron_to_mesh(file_path, output_obj_path, output_format="obj", apply_binary_closing=True, apply_gaussian_filter=True, fix_gaps_x_axis=True):
    with h5py.File(file_path, "r") as f:
        dataset = f["main"]
        mask_shape = dataset.shape
        print(f"Dataset shape: {mask_shape}")
        mask = dataset[:].astype(np.uint8)

    print("Preprocessing mask...")
    if apply_binary_closing:
        print("Applying binary closing...")
        mask = binary_closing(mask, structure=np.ones((3, 3, 3))).astype(np.uint8)
    
    if fix_gaps_x_axis:
        print("Fixing gaps along the X-axis (interpolation)...")
        x_vals = np.arange(mask.shape[0])
        valid_slices = np.any(mask, axis=(1, 2))
        if np.any(valid_slices):
            if np.sum(valid_slices) < 2 and 'linear' in 'linear':
                 print("Warning: Less than 2 valid slices found for X-axis interpolation with linear kind. Skipping this step to avoid error.")
            else:
                interp_func = interp1d(x_vals[valid_slices], mask[valid_slices], axis=0, kind='linear', fill_value="extrapolate")
                mask = interp_func(x_vals).astype(np.uint8)
        else:
            print("Warning: No valid slices found for X-axis interpolation. Skipping this step.")

    if apply_gaussian_filter:
        print("Applying Gaussian filter...")
        mask = gaussian_filter(mask.astype(float), sigma=1) > 0.5
        mask = mask.astype(np.uint8)
    else:
        mask = mask.astype(bool).astype(np.uint8)


    print("Running Marching Cubes to extract surface mesh...")
    verts, faces, _, _ = marching_cubes(mask, level=0.5)
    if verts.size == 0 or faces.size == 0:
        print("Warning: Marching cubes resulted in an empty mesh. This might be due to an empty or unsuitable mask after preprocessing.")
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    if not mesh.is_empty:
        print("Filling holes in the mesh...")
        filled_mesh_candidate = mesh.fill_holes()
        if isinstance(filled_mesh_candidate, trimesh.Trimesh):
            mesh = filled_mesh_candidate
        else:
            print(f"Warning: Hole filling did not return a valid mesh object (type: {type(filled_mesh_candidate)}). Proceeding with the mesh before hole filling.")
    else:
        print("Mesh is empty, skipping hole filling.")


    try:
        supported_formats = set(trimesh.exchange.export.mesh_formats())
    except AttributeError:
        supported_formats = {"obj", "stl", "ply", "glb", "gltf", "collada", "dae", "off", "xyz", "json", "dict", "dict64", "msgpack"}
        print("Warning: Could not dynamically fetch supported formats from trimesh. Using a predefined list.")

    if output_format.lower() not in supported_formats:
        print(f"Error: Invalid output format '{output_format}'.")
        print(f"Supported formats are: {', '.join(sorted(list(supported_formats)))}")
        return

    print(f"Saving {output_format.upper()} file to {output_obj_path}...")
    try:
        mesh.export(output_obj_path, file_type=output_format.lower())
        print(f"{output_format.upper()} Export Complete!")
    except Exception as e:
        print(f"Error exporting mesh to '{output_obj_path}' as {output_format.upper()}: {e}")
        print("Please ensure the output path is valid, writable, and the format is correctly supported by your trimesh installation.")

