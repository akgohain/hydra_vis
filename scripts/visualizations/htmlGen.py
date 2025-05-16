import os
import json
import shutil
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

TEMPLATE_HTML_PATH = "vesicle_template.html"

def convert_parquet_to_json(parquet_path, output_path):
    """
        Convert Parquet to JSON

        This function converts a Parquet file to a JSON file.

        Args:
            parquet_path (str): The path to the input Parquet file.
            output_path (str): The path to the output JSON file.

        Returns:
            None
    """
    df = pd.read_parquet(parquet_path)
    vesicles = df.to_dict(orient="records")
    with open(output_path, 'w') as f:
        json.dump(vesicles, f)
    print(f"Converted vesicle data to JSON: {output_path}")

def parse_offset_csv(csv_path):
    """
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
    """
    offsets = {}
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            neuron, z_min, z_max, y_min, y_max, x_min, x_max = parts
            x_offset = int(x_min)
            y_offset = int(y_min)
            z_offset = int(z_min)
            if neuron.strip() == "SHL17":
                y_offset += 4000
            offsets[neuron.strip()] = {"x": x_offset, "y": y_offset, "z": z_offset}
    return offsets

def generate_html(
    vesicle_parquet_path: str,
    offset_csv_path: str,
    neuron_glb_dir: str,
    vesicle_color_map_path: str,
    output_dir: str = "vesicle_viewer_output"
):
    """
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
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    neuron_output_dir = output_dir / "neurons"

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    neuron_output_dir.mkdir(parents=True, exist_ok=True)

    vesicle_json_path = data_dir / "vesicles.json"
    convert_parquet_to_json(vesicle_parquet_path, vesicle_json_path)

    color_map_output_path = data_dir / "colormap.json"
    shutil.copy(vesicle_color_map_path, color_map_output_path)
    print(f"Custom color map copied to: {color_map_output_path}")

    neuron_files = list(Path(neuron_glb_dir).glob("*.glb"))
    for file in neuron_files:
        shutil.copy(file, neuron_output_dir / file.name)
    print(f"Copied {len(neuron_files)} neuron GLB files to: {neuron_output_dir}")

    with open(TEMPLATE_HTML_PATH, 'r') as template_file:
        html_content = template_file.read()

    html_output_path = output_dir / "index.html"
    with open(html_output_path, 'w') as out_html:
        out_html.write(html_content)
    print(f"Viewer HTML created at: {html_output_path}")