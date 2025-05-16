import polars as pl
import trimesh
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from matplotlib import cm
from matplotlib import colors as mcolors

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
        return np.tile([200, 200, 200, 255], (df.height, 1))  # default gray

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


def vesicles_to_plotly(
    parquet_path,
    neuron_mesh_path=None,
    filter_sample_id=None,
    color_by=None,
    colormap="viridis",
    output_html_path=None,
    marker_size=3,
    verbose=True,
    neuron_opacity=0.3
):
    """
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
    """
    parquet_path = Path(parquet_path)

    if verbose:
        print(f"Loading vesicle data from {parquet_path}...")
    df = pl.read_parquet(parquet_path)

    if filter_sample_id:
        df = df.filter(pl.col("sample_id") == filter_sample_id)
        if verbose:
            print(f"🔍 Filtering by sample_id: {filter_sample_id} → {len(df)} vesicles")

    required_cols = {"x", "y", "z"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Parquet must contain at least these columns: {required_cols}")

    data = []
    if neuron_mesh_path:
        neuron_mesh_path = Path(neuron_mesh_path)
        if verbose:
            print(f"Loading neuron mesh from {neuron_mesh_path}...")
        neuron_mesh = trimesh.load(neuron_mesh_path, process=False)

        if neuron_mesh.is_empty:
            raise ValueError("Neuron mesh is empty.")

        neuron_trace = go.Mesh3d(
            x=neuron_mesh.vertices[:, 0],
            y=neuron_mesh.vertices[:, 1],
            z=neuron_mesh.vertices[:, 2],
            i=neuron_mesh.faces[:, 0],
            j=neuron_mesh.faces[:, 1],
            k=neuron_mesh.faces[:, 2],
            color='lightgray',
            opacity=neuron_opacity,
            name="Neuron Mesh",
            showscale=False
        )
        data.append(neuron_trace)

    if verbose:
        print(f"Computing colors using {colormap} over {color_by or '[default gray]'}")
    colors = compute_vertex_colors(df, color_by=color_by, colormap=colormap)

    scatter_trace = go.Scatter3d(
        x=df["x"],
        y=df["y"],
        z=df["z"],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=["rgba({}, {}, {}, {})".format(r, g, b, a/255) for r, g, b, a in colors],
            opacity=0.85
        ),
        name="Vesicles",
        text=df["vesicle_id"] if "vesicle_id" in df.columns else None,
        hoverinfo='text'
    )
    data.append(scatter_trace)

    fig = go.Figure(data=data)
    fig.update_layout(
        scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data'),
        title=f"Vesicles on Neuron Mesh ({filter_sample_id})" if filter_sample_id else "Vesicles on Neuron Mesh",
        legend_title="Legend"
    )

    if output_html_path:
        fig.write_html(output_html_path)
        if verbose:
            print(f"Saved interactive plot to {output_html_path}")
    else:
        fig.show()

    return fig