import os
import csv
import trimesh
import numpy as np
import pyvista as pv

# TODO: logging with tqdm
# TODO: generalize beyond obj

def load_offsets(csv_path):
    offsets = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['name']
            offsets[name] = np.array([int(row['x']), int(row['y']), int(row['z'])])
    return offsets

def load_trimesh_objs(folder):
    meshes = {}
    for filename in os.listdir(folder):
        if filename.endswith('.obj'):
            name = os.path.splitext(filename)[0]
            path = os.path.join(folder, filename)
            meshes[name] = trimesh.load(path, process=False)
    return meshes

def load_vesicle_trimesh_objs(folder):
    vesicles = {}
    for filename in os.listdir(folder):
        if filename.endswith('.obj'):
            neuron_id = filename.split('_')[0]
            path = os.path.join(folder, filename)
            mesh = trimesh.load(path, process=False)
            vesicles.setdefault(neuron_id, []).append(mesh)
    return vesicles

def trimesh_to_pyvista(mesh):
    """Convert a trimesh mesh to a PyVista PolyData with vertex colors if available"""
    faces = np.hstack([
        np.full((mesh.faces.shape[0], 1), 3),
        mesh.faces
    ]).astype(np.int64)
    pd_mesh = pv.PolyData(mesh.vertices, faces=faces)

    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        pd_mesh.point_data['Colors'] = colors
    return pd_mesh

def render_scene(neuron_mesh_dir, vesicle_mesh_dir, offsets_csv):
    offsets = load_offsets(offsets_csv)

    neurons = load_trimesh_objs(neuron_mesh_dir)

    vesicles = load_vesicle_trimesh_objs(vesicle_mesh_dir)

    plotter = pv.Plotter()

    for name, mesh in neurons.items():
        offset = offsets.get(name, np.array([0, 0, 0]))
        mesh.apply_translation(offset)
        pd_mesh = trimesh_to_pyvista(mesh)
        plotter.add_mesh(pd_mesh, name=f'neuron_{name}', show_scalar_bar=False)

    for neuron_id, vesicle_list in vesicles.items():
        offset = offsets.get(neuron_id, np.array([0, 0, 0]))
        for vmesh in vesicle_list:
            vmesh.apply_translation(offset)
            pd_vmesh = trimesh_to_pyvista(vmesh)
            plotter.add_mesh(pd_vmesh, name=f'vesicle_{neuron_id}', show_scalar_bar=False)

    plotter.show()