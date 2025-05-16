# imports here for now during reorg
import h5py
import numpy as np
import polars as pl
import trimesh
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing, gaussian_filter
from skimage.measure import marching_cubes
from matplotlib import cm
from matplotlib import colors as mcolors

# TODO: implement TQDM logging
# from tqdm import tqdm

def neuron_to_mesh(file_path, output_obj_path, output_format="obj", apply_binary_closing=True, apply_gaussian_filter=True, fix_gaps_x_axis=True):
    """
    Converts a neuron mask from an HDF5 file to a 3D mesh and saves it in a specified format.
    This function takes a file path to an HDF5 file containing a neuron mask, preprocesses the mask,
    generates a mesh using marching cubes, and saves the mesh to a specified output path in a given format.
    Args:
        file_path (str): Path to the HDF5 file containing the neuron mask. The HDF5 file should contain a dataset named "main".
        output_obj_path (str): Path to save the generated mesh file.
        output_format (str, optional): The format to save the mesh in (e.g., "obj", "stl", "ply"). Defaults to "obj".
        apply_binary_closing (bool, optional): Whether to apply binary closing to the mask. Defaults to True.
        apply_gaussian_filter (bool, optional): Whether to apply a Gaussian filter to the mask. Defaults to True.
        fix_gaps_x_axis (bool, optional): Whether to fix gaps along the X-axis using linear interpolation. Defaults to True.
    Returns:
        None: The function saves a mesh file to the specified output path.  Prints status and error messages to the console.
    """
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

def vesicles_to_mesh(
    parquet_path,
    output_path,
    output_format="obj",
    sphere_resolution=4,
    color_by=None,
    colormap="viridis",
    verbose=True
):
    """
    Generates a mesh of vesicles from a parquet file and saves it.
    
    Parameters:
    - parquet_path (str or Path): Path to .parquet with 'x', 'y', 'z', 'radius' and optional data columns.
    - output_path (str or Path): Path to save the output mesh file.
    - output_format (str): Desired output format (e.g., "obj", "ply", "stl").
    - sphere_resolution (int): Level of icosphere detail per vesicle.
    - color_by (str or list of str, optional): Column(s) in the Parquet to base vertex color on.
    - colormap (str or callable): Matplotlib colormap name or custom [0,1] -> RGBA callable.
    - verbose (bool): If True, print progress.

    Returns:
    - None. The mesh is saved to the specified output_path.
    """
    parquet_path = Path(parquet_path)
    df = pl.read_parquet(parquet_path)

    required_cols = {"x", "y", "z", "radius"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Parquet must contain at least these columns: {required_cols}")

    if verbose:
        print(f"Loaded {len(df)} vesicles from {parquet_path.name}")

    coords = df.select(["x", "y", "z"]).to_numpy()
    radii = df.select("radius").to_numpy().flatten()

    colors = compute_vertex_colors(df, color_by=color_by, colormap=colormap)

    sphere_meshes = []
    for i, (center, r, color) in enumerate(zip(coords, radii, colors)):
        sphere = trimesh.creation.icosphere(subdivisions=sphere_resolution, radius=r)
        sphere.apply_translation(center)
        sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1))
        sphere_meshes.append(sphere)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  🎨 Processed {i+1}/{len(coords)} vesicles...")

    if not sphere_meshes:
        print("Warning: No vesicles processed, resulting mesh will be empty.")
        combined_mesh = trimesh.Trimesh()
    else:
        combined_mesh = trimesh.util.concatenate(sphere_meshes)

    if verbose:
        print(f"Combined mesh: {len(combined_mesh.vertices)} vertices, {len(combined_mesh.faces)} faces")

    try:
        supported_formats = set(trimesh.exchange.export.mesh_formats())
    except AttributeError:
        supported_formats = {"obj", "stl", "ply", "glb", "gltf", "collada", "dae", "off", "xyz", "json", "dict", "dict64", "msgpack"}
        if verbose:
            print("Warning: Could not dynamically fetch supported formats from trimesh. Using a predefined list.")

    if output_format.lower() not in supported_formats:
        print(f"Error: Invalid output format '{output_format}'.")
        print(f"Supported formats are: {', '.join(sorted(list(supported_formats)))}")
        return

    if combined_mesh.is_empty:
        print(f"Warning: The generated mesh is empty. Skipping export to {output_path}.")
        return

    print(f"Saving {output_format.upper()} file to {output_path}...")
    try:
        combined_mesh.export(output_path, file_type=output_format.lower())
        print(f"{output_format.upper()} Export Complete!")
    except Exception as e:
        print(f"Error exporting mesh to '{output_path}' as {output_format.upper()}: {e}")
        print("Please ensure the output path is valid, writable, and the format is correctly supported by your trimesh installation.")

def compute_vertex_colors(df, color_by=None, colormap="viridis"):
    """
    Computes per-row RGB colors based on one or more DataFrame columns.

    Parameters:
    - df (pl.DataFrame): DataFrame with vesicle data.
    - color_by (str or list of str): Column(s) to derive color from.
    - colormap (str or callable): Matplotlib colormap name or function.

    Returns:
    - colors (np.ndarray): (N, 4) array of RGBA values (uint8).
    """
    if color_by is None:
        return np.tile([200, 200, 200, 255], (df.height, 1))

    if isinstance(color_by, str):
        values = df[color_by].to_numpy()
    elif isinstance(color_by, list):
        values = df.select(color_by).to_numpy().sum(axis=1)
    else:
        raise ValueError("color_by must be a string or list of strings")

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap(colormap) if isinstance(colormap, str) else colormap
    rgba = cmap(norm(values))
    rgb255 = (rgba * 255).astype(np.uint8)
    return rgb255