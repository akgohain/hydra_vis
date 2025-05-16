# TODO: tqdm logging
# TODO: example txt files in readme

from pathlib import Path
import re
import polars as pl
import numpy as np
from scipy.spatial import KDTree

def extract_vesicle_data(
    input_path,
    output_path="vesicle_com_data.parquet",
    types_dir=None,
    compute_neighbors=False,
    neighbor_radius_nm=500.0,
    verbose=True
):
    """
    Parses vesicle mapping file(s) to extract center-of-mass (COM) and metadata,
    swaps X and Z coordinates, optionally joins type labels from a directory,
    and optionally computes neighbor densities within a specified radius.

    Parameters:
    - input_path (str or Path): File or directory with *_mapping.txt files.
    - output_path (str): Where to save result (.parquet, .csv, .json).
    - types_dir (str or Path or None): Directory with *_label.txt files (optional).
    - compute_neighbors (bool): Whether to compute neighbor densities (default: False).
    - neighbor_radius_nm (float): Radius in nanometers for KDTree neighbor ball (default: 500).
    - verbose (bool): Print status and preview (default: True).

    ⚠️ Assumes voxel physical dimensions are (x=30nm, y=8nm, z=8nm). These may vary by dataset.

    Returns:
    - df (pl.DataFrame): Extracted DataFrame with optional neighbor info.
    """
    input_path = Path(input_path).resolve()
    output_file = Path(output_path).resolve()
    all_files = []

    if input_path.is_file() and input_path.name.endswith("_mapping.txt"):
        all_files = [input_path]
    elif input_path.is_dir():
        all_files = list(input_path.glob("*_mapping.txt"))
    else:
        raise ValueError("Input path must be a *_mapping.txt file or a directory of them.")

    records = []
    pattern = re.compile(
        r'\((\d+\.?\d*), (\d+\.?\d*), (\d+\.?\d*)\): \(\'(\w+)\', (\w+_\d+), (\d+), (\d+\.?\d*(?:e[-+]?\d+)?)'
    )

    for file_path in all_files:
        vesicle_type = "lv" if "_lv_" in file_path.name else "sv"
        sample_id = file_path.stem.split('_')[0]

        with open(file_path, 'r') as f:
            content = f.read()

        matches = pattern.findall(content)

        for match in matches:
            x, y, z = float(match[0]), float(match[1]), float(match[2])
            _, v_id, volume, radius = match[3], match[4], int(match[5]), float(match[6])
            records.append({
                "sample_id": sample_id,
                "vesicle_type": vesicle_type,
                "vesicle_id": v_id,
                "x": x,
                "y": y,
                "z": z,
                "volume": volume,
                "radius": radius
            })

    if not records:
        raise ValueError("No valid vesicle entries found.")

    df = pl.DataFrame(records)

    df = df.rename({"x": "temp_x", "z": "x"})
    df = df.rename({"temp_x": "z"})

    if types_dir:
        types_dir = Path(types_dir).resolve()
        label_files = list(types_dir.glob("*_lv_label.txt")) + list(types_dir.glob("*_sv_label.txt"))
        type_records = []

        type_pattern = re.compile(r'\((\d+):(\d+)\)')

        for file_path in label_files:
            vesicle_prefix = "lv" if "_lv_" in file_path.name else "sv"
            sample_id = file_path.stem.split('_')[0]

            with open(file_path, 'r') as f:
                content = f.read()

            matches = type_pattern.findall(content)

            for vesicle_num, type_val in matches:
                vesicle_id = f"{vesicle_prefix}_{vesicle_num}"
                type_records.append({
                    "sample_id": sample_id,
                    "vesicle_id": vesicle_id,
                    "type": int(type_val)
                })

        df_types = pl.DataFrame(type_records)
        df = df.join(df_types, on=["sample_id", "vesicle_id"], how="left")
        df = df.with_columns(pl.col("type").fill_null(0))

    if compute_neighbors:
        if verbose:
            print(f"Computing neighbor counts within {neighbor_radius_nm}nm...")

        df = df.with_row_count("_row_id")

        voxel_dims = np.array([30, 8, 8])
        all_counts = {}

        for sample_id in df["sample_id"].unique().to_list():
            sample_df = df.filter(pl.col("sample_id") == sample_id)
            coords = sample_df.select(["x", "y", "z"]).to_numpy() * voxel_dims
            row_ids = sample_df["_row_id"].to_numpy()

            if len(coords) == 0:
                continue

            tree = KDTree(coords)
            neighbors = tree.query_ball_point(coords, r=neighbor_radius_nm)

            for i, ids in enumerate(neighbors):
                all_counts[row_ids[i]] = len(ids) - 1

        counts_list = [all_counts.get(i, 0) for i in range(df.height)]
        df = df.with_columns(pl.Series("neighbors_within_radius", counts_list))
        df = df.drop("_row_id")

    ext = output_file.suffix.lower()
    if ext == ".csv":
        df.write_csv(output_file)
    elif ext == ".json":
        df.write_json(output_file, row_oriented=True)
    else:
        if ext != ".parquet":
            output_file = output_file.with_suffix(".parquet")
            if verbose:
                print(f"Unsupported extension. Saving as Parquet: {output_file}")
        df.write_parquet(output_file)

    if verbose:
        print(f"Data saved to: {output_file}")
        print(df.head())

    return df