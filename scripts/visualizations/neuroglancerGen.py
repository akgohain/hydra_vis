import os
import h5py
import gc
import csv
import socket
import numpy as np
import neuroglancer
import logging
import yaml
from contextlib import closing

# TODO: improve logging via tqdm
# TODO improve name regex to not require hardcoded name structure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def append_seg_layer(viewer_txn, name, data, res, offset):
    if data is not None:
        viewer_txn.layers.append(
            name=name,
            layer=ng_seg_layer(data, res, offset)
        )

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def load_offsets_csv(csv_path):
    offsets = {}
    try:
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                offsets[name] = [int(row['z']), int(row['y']), int(row['x'])]
    except FileNotFoundError:
        logging.error(f"Offset file not found: {csv_path}")
        return None
    except KeyError as e:
        logging.error(f"Missing column in offset file: {e}")
        return None
    return offsets

def load_h5_volume(path, key):
    try:
        with h5py.File(path, 'r') as f:
            return np.array(f[key])
    except FileNotFoundError:
        logging.error(f"HDF5 file not found: {path}")
        return None
    except KeyError:
        logging.error(f"Key not found in HDF5 file: {key} in {path}")
        return None

def ng_seg_layer(data, res, offset):
    return neuroglancer.LocalVolume(
        data.astype(np.uint32),
        dimensions=neuroglancer.CoordinateSpace(
            names=["z", "y", "x"],
            units=["nm", "nm", "nm"],
            scales=res
        ),
        voxel_offset=offset,
        volume_type='segmentation'
    )

def launch_neuroglancer(neuron_h5_dir, vesicle_h5_path, offset_csv_path, voxel_resolution=(30, 64, 64)):
    ip = 'localhost'
    port = find_free_port()
    neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
    viewer = neuroglancer.Viewer()

    offsets = load_offsets_csv(offset_csv_path)
    if offsets is None:
        logging.error("Failed to load offsets, exiting")
        return None

    try:
        with h5py.File(vesicle_h5_path, 'r') as vesicle_h5:
            with viewer.txn() as s:
                for fname in os.listdir(neuron_h5_dir):
                    if not fname.endswith('.h5'):
                        continue

                    name = os.path.splitext(fname)[0]
                    if name not in offsets or name not in vesicle_h5:
                        logging.warning(f"Skipping {name}: missing offset or vesicle data")
                        continue

                    offset = offsets[name]
                    neuron_path = os.path.join(neuron_h5_dir, fname)
                    neuron_data = load_h5_volume(neuron_path, "main")
                    if neuron_data is None:
                        logging.error(f"Failed to load neuron data for {name}, skipping")
                        continue
                    
                    try:
                        vesicle_data = np.array(vesicle_h5[name])
                    except KeyError:
                        logging.error(f"Failed to load vesicle data for {name}, skipping")
                        continue

                    append_seg_layer(s, f'neuron_{name}', neuron_data, voxel_resolution, offset)
                    append_seg_layer(s, f'vesicles_{name}', vesicle_data, voxel_resolution, offset)

                    logging.info(f"Loaded {name} neuron & vesicles into viewer")
                    del neuron_data, vesicle_data
                    gc.collect()

    except FileNotFoundError:
        logging.error(f"Vesicle HDF5 file not found: {vesicle_h5_path}")
        return None

    print(" Neuroglancer viewer is live:")
    print(viewer)
    return viewer