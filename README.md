# ves_vis

## neuroglancerGen.py

### `append_seg_layer`
```
Appends a segmentation layer to a Neuroglancer viewer transaction.

Args:
    viewer_txn: The Neuroglancer viewer transaction to append the layer to.
    name: The name of the layer.
    data: The segmentation data.
    res: The resolution of the data.
    offset: The offset of the data.
```

### `find_free_port`
```
Finds a free port on the system.

This function creates a socket, binds it to a random port, and then returns the port number.
This can be useful for applications that need to listen on a port but don't want to hardcode a specific port number.

Returns:
    int: A free port number on the system.
```

### `load_offsets_csv`
```
Loads offsets from a CSV file into a dictionary.

The CSV file should have columns 'name', 'z', 'y', and 'x'.
The 'name' column is the key in the dictionary, and the 'z', 'y', and 'x' columns are the values (as a list of integers).

Args:
    csv_path (str): The path to the CSV file.

Returns:
    dict: A dictionary where keys are names and values are lists of [z, y, x] offsets.
          Returns None if the file is not found or if there's a missing column.
```

### `load_h5_volume`
```
Loads a volume from an HDF5 file.

Args:
    path (str): The path to the HDF5 file.
    key (str): The key of the volume to load.

Returns:
    np.ndarray: The volume as a NumPy array, or None if the file or key is not found.
```

### `ng_seg_layer`
```
Generates a Neuroglancer segmentation layer from a data array.

Args:
    data (np.ndarray): The segmentation data as a NumPy array.  Should be convertible to np.uint32.
    res (list or tuple): A list or tuple of three numbers representing the resolution (in nm) of the data in z, y, and x dimensions.
    offset (list or tuple): A list or tuple of three numbers representing the offset of the data in z, y, and x dimensions.

Returns:
    neuroglancer.LocalVolume: A Neuroglancer LocalVolume object configured for segmentation.
```

### `launch_neuroglancer`
```
Launches a Neuroglancer viewer to visualize neuron and vesicle data.

Args:
    neuron_h5_dir (str): Path to the directory containing neuron HDF5 files.
        Each HDF5 file should contain a 'main' dataset representing the neuron volume.
    vesicle_h5_path (str): Path to the HDF5 file containing vesicle data.
        Each dataset within this file should correspond to a neuron's name and
        contain the vesicle volume data.
    offset_csv_path (str): Path to the CSV file containing offset information for each neuron.
        The CSV should have a column for neuron names and corresponding x, y, z offsets.
    voxel_resolution (tuple, optional): Voxel resolution in (z, y, x) order.
        Defaults to (30, 64, 64).
Returns:
    neuroglancer.Viewer: The Neuroglancer viewer object if successful, None otherwise.
        The viewer will display the neuron and vesicle data as segmentation layers,
        with appropriate offsets applied.
Raises:
    FileNotFoundError: If the vesicle HDF5 file is not found.
    KeyError: If a neuron name in the neuron directory or offset CSV is not found
        as a dataset in the vesicle HDF5 file.
Notes:
    - The function loads neuron and vesicle data, creates segmentation layers in Neuroglancer,
        and applies offsets to align the data.
    - It iterates through the HDF5 files in the specified neuron directory.
    - It expects the vesicle HDF5 file to contain datasets named after the neurons.
    - It uses a CSV file to determine the spatial offset for each neuron.
    - The function attempts to handle missing data gracefully by skipping neurons
        with missing offset or vesicle data.
    - It explicitly deletes neuron and vesicle data and calls garbage collection
        to manage memory usage.
```

## pyvistaGen.py

### `load_offsets`
```
Loads offsets from a CSV file into a dictionary.

The CSV file should have columns 'name', 'x', 'y', and 'z'.
The 'name' column is used as the key in the dictionary, and the
'x', 'y', and 'z' columns are used to create a NumPy array of
offsets.

Args:
    csv_path (str): The path to the CSV file.

Returns:
    dict: A dictionary where the keys are the names from the CSV
        file and the values are NumPy arrays representing the
        offsets.
```

### `load_trimesh_objs`
```
Loads all .obj files from a given folder into a dictionary of trimesh objects.

Args:
    folder (str): The path to the folder containing the .obj files.

Returns:
    dict: A dictionary where keys are the names of the .obj files (without extension)
            and values are the corresponding trimesh objects.
            Returns an empty dictionary if no .obj files are found.
```

### `load_vesicle_trimesh_objs`
```
Loads vesicle meshes from OBJ files in a folder, grouping them by neuron ID.

Args:
    folder (str): Path to the folder containing the OBJ files.  Filenames
        are expected to start with the neuron ID, followed by an underscore.

Returns:
    dict: A dictionary where keys are neuron IDs and values are lists of
        trimesh.Trimesh objects representing the vesicles for that neuron.
        Returns an empty dictionary if no OBJ files are found.
```

### `trimesh_to_pyvista`
```
Convert a trimesh mesh to a PyVista PolyData object, handling vertex colors.

    This function takes a trimesh.Trimesh object as input and converts it into a
    PyVista PolyData object suitable for visualization and further processing within
    the PyVista ecosystem.  It correctly handles the mesh's vertices and faces,
    and crucially, it also transfers vertex color information if it is present in
    the trimesh object.

    Args:
        mesh (trimesh.Trimesh): The input trimesh.Trimesh object to convert.  This
            object is expected to contain vertex and face data, and optionally,
            vertex color data accessible via `mesh.visual.vertex_colors`.

    Returns:
        pyvista.PolyData: A PyVista PolyData object representing the converted mesh.
            The PolyData object contains the mesh's vertices and faces. If the input
            trimesh object has vertex colors, these are also included in the
            PolyData as a point data array named 'Colors'.

    Raises:
        AttributeError: If the input `mesh` does not have the expected attributes
            (e.g., `vertices`, `faces`, or `visual`).
        TypeError: If the vertex color data is not of the expected type (e.g., not a
            NumPy array) or if the face data is not convertible to int64.
        ValueError: If the vertex color data has an unexpected shape (e.g., not an
            Nx3 or Nx4 array).  If an Nx4 array is provided, the alpha channel is
            discarded.
```

### `render_scene`
```
Renders a scene composed of neuron and vesicle meshes using PyVista.
This function loads neuron and vesicle meshes from specified directories,
applies positional offsets, and visualizes them in a PyVista plotter.
It supports interactive display or saving a screenshot of the scene.
Args:
    neuron_mesh_dir (str): Path to the directory containing neuron mesh files in OBJ format.
                                The filenames (without extension) are used as neuron IDs.
    vesicle_mesh_dir (str): Path to the directory containing vesicle mesh files in OBJ format.
                                Vesicles are expected to be organized in subdirectories named after
                                the neuron ID they belong to.
    offsets_csv (str): Path to a CSV file containing positional offsets for each neuron.
                            The CSV should have columns 'id', 'x', 'y', and 'z'.
                            If an ID is not found in the CSV, a default offset of [0, 0, 0] is used.
    show_plotter (bool, optional): If True, displays the scene in an interactive PyVista plotter window.
                                    If False, renders the scene off-screen and saves a screenshot.
                                    Defaults to True.
    camera_position (list or str, optional):  The camera position to use for the plot.  Can be a list of
                                                three points defining the camera position, focal point,
                                                and view-up direction, or a string such as 'xy', 'xz', etc.
                                                Defaults to None, which uses the PyVista default camera position.
    screenshot_path (str, optional): Path to save the screenshot of the scene if `show_plotter` is False.
                                        Defaults to 'scene.png'.
Returns:
    None
Raises:
    FileNotFoundError: If the specified neuron or vesicle mesh directories do not exist.
    ValueError: If the offsets CSV file is not properly formatted.
Notes:
    - The function assumes that neuron mesh files have the .obj extension.
    - Vesicle mesh directories should contain .obj files directly.
    - The offsets CSV file should have a header row.
    - The function uses `trimesh` for loading meshes and `pyvista` for rendering.
    - If `show_plotter` is False, the function runs in off-screen mode, which requires a properly configured
        environment for off-screen rendering.
Examples:
    Render the scene with default settings:
    ```python
    render_scene('neurons/', 'vesicles/', 'offsets.csv')
    ```
    Render the scene and save a screenshot:
    ```python
    render_scene('neurons/', 'vesicles/', 'offsets.csv', show_plotter=False, screenshot_path='my_scene.png')
    ```
    Render the scene with a specific camera position:
    ```python
    camera_pos = [(10, 10, 10), (0, 0, 0), (0, 1, 0)]  # Camera position, focal point, view-up
    render_scene('neurons/', 'vesicles/', 'offsets.csv', camera_position=camera_pos)
    ```
```

## plotlyGen.py

### `compute_vertex_colors`
```
Computes per-row RGB colors based on one or more DataFrame columns.

Parameters:
- df (pl.DataFrame): DataFrame with vesicle data.
- color_by (str or list of str): Column(s) to derive color from.
- colormap (str or callable): Matplotlib colormap name or function.

Returns:
- colors (np.ndarray): (N, 4) array of RGBA values (uint8).
```

### `vesicles_to_plotly`
```
Loads a neuron mesh and vesicle COMs from a parquet file, optionally filters by sample,
and generates a 3D Plotly visualization with overlaid vesicle markers colored by a colormap.

Parameters:
- parquet_path (str or Path): Parquet file with x, y, z, radius and optional metadata.
- neuron_mesh_path (str or Path, optional): Path to neuron mesh file (.obj, .ply, .glb, etc).
- filter_sample_id (str, optional): If specified, filters vesicles by sample_id.
- color_by (str or list of str): Column(s) in Parquet to color vesicles by.
- colormap (str or callable): Colormap used to color points.
- output_html_path (str or Path, optional): Saves the interactive plot to this path.
- marker_size (int): Marker size for vesicles.
- verbose (bool): Print progress updates.
- neuron_opacity (float): Opacity of the neuron mesh.

Returns:
- fig (plotly.graph_objects.Figure): The interactive Plotly figure.
```

## htmlGen.py

### `convert_parquet_to_json`
```
Convert Parquet to JSON

This function converts a Parquet file to a JSON file.

Args:
    parquet_path (str): The path to the input Parquet file.
    output_path (str): The path to the output JSON file.

Returns:
    None
```

### `parse_offset_csv`
```
Parses a CSV file to extract neuron offsets.

Args:
    csv_path (str): The path to the CSV file containing neuron offset data.
        The CSV should have the format: neuron, z_min, z_max, y_min, y_max, x_min, x_max, ... (other columns are ignored)
        where:
            - neuron (str): The name of the neuron.
            - z_min (int): The minimum Z coordinate.
            - z_max (int): The maximum Z coordinate.
            - y_min (int): The minimum Y coordinate.
            - y_max (int): The maximum Y coordinate.
            - x_min (int): The minimum X coordinate.
            - x_max (int): The maximum X coordinate.

Returns:
    dict: A dictionary where keys are neuron names (str) and values are dictionaries
        containing the x, y, and z offsets (int).  For example:
        {'neuron_A': {'x': 10, 'y': 20, 'z': 30}, 'neuron_B': {'x': 40, 'y': 50, 'z': 60}}
        Note: Applies a +4000 offset to the y coordinate of neuron "SHL17".
```

### `generate_html`
```
Generates an HTML file for visualizing vesicles and neurons.
This function takes paths to vesicle data, neuron models, and a color map,
and generates an HTML file that can be used to visualize the data in a web browser.
It also copies the vesicle data, neuron models, and color map to the output directory.
Args:
    vesicle_parquet_path (str): Path to the Parquet file containing vesicle data.
    offset_csv_path (str): Path to the CSV file containing offset data (currently unused).
    neuron_glb_dir (str): Path to the directory containing neuron GLB files.
    vesicle_color_map_path (str): Path to the JSON file containing the vesicle color map.
    output_dir (str, optional): Path to the output directory. Defaults to "vesicle_viewer_output".
Returns:
    None: This function does not return any value. It generates files in the specified output directory.
```

## meshGen.py

### `neuron_to_mesh`
```
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
```

### `vesicles_to_mesh`
```
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
```

### `compute_vertex_colors`
```
Computes per-row RGB colors based on one or more DataFrame columns.

Parameters:
- df (pl.DataFrame): DataFrame with vesicle data.
- color_by (str or list of str): Column(s) to derive color from.
- colormap (str or callable): Matplotlib colormap name or function.

Returns:
- colors (np.ndarray): (N, 4) array of RGBA values (uint8).
```

## dfGen.py

### `extract_vesicle_data`
```
Parses vesicle mapping file(s) to extract center-of-mass (COM) and metadata,
swaps X and Z coordinates, optionally joins type labels from a directory,
and optionally computes neighbor densities within a specified radius.

Parameters:
- input_path (str or Path): File or directory with *_mapping.txt files.
- output_path (str): Where to save result (.parquet, .csv, .json).
- types_dir (str or Path or None): Directory with *_label.txt files (optional).
- compute_neighbors (bool): Whether to compute neighbor densities (default: False).
- neighbor_radius_nm (float): Radius in nanometers for KDTree neighbor ball (default: 500).
- voxel_dims_nm (tuple or list): Physical dimensions (x,y,z) of a voxel in nanometers.
                                 Defaults to (30, 8, 8). Note that the function
                                 internally swaps X and Z, so for KDTree calculations,
                                 the order applied to coordinates (z_swapped, y, x_swapped)
                                 will effectively be (voxel_x, voxel_y, voxel_z).
- verbose (bool): Print status and preview (default: True).

Assumes voxel physical dimensions are (x=30nm, y=8nm, z=8nm) by default.
These may vary by dataset. Ensure `voxel_dims_nm` is set correctly if your data differs.

Returns:
- df (pl.DataFrame): Extracted DataFrame with optional neighbor info.
```