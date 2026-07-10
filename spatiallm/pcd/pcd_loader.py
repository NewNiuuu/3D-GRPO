import logging

import numpy as np
import open3d as o3d

from .blob_utils import has_blob_config, should_use_blob, read_blob_media_bytes

log = logging.getLogger(__name__)

# Mapping from PLY property types to little-endian numpy dtypes.
_PLY_NP_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def _parse_binary_ply_bytes(data: bytes) -> "o3d.geometry.PointCloud":
    """Parse a binary-little-endian PLY from raw bytes, fully in memory.

    Open3D's own `read_point_cloud_from_bytes` only supports the `mem::xyz`
    format (no PLY/PCD), so we parse the PLY vertex block directly with numpy:
    read the header, build a structured dtype, `np.frombuffer` the vertices, and
    assemble an Open3D PointCloud whose points/colors match
    `o3d.io.read_point_cloud` on the same file. Nothing is written to disk.
    """
    if not data.startswith(b"ply"):
        raise ValueError("Not a PLY file.")

    marker = data.index(b"end_header")
    header_end = data.index(b"\n", marker) + 1
    header_lines = data[:header_end].decode("ascii").splitlines()

    fmt = next(l for l in header_lines if l.startswith("format")).split()[1]
    if fmt != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {fmt}.")

    num_vertices = None
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("element"):
            parts = line.split()
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                num_vertices = int(parts[2])
        elif line.startswith("property") and in_vertex:
            parts = line.split()
            if parts[1] == "list":
                raise ValueError("List properties are not supported.")
            props.append((parts[2], "<" + _PLY_NP_TYPES[parts[1]]))

    if num_vertices is None:
        raise ValueError("No vertex element found in PLY header.")

    dtype = np.dtype(props)
    block = data[header_end:header_end + num_vertices * dtype.itemsize]
    arr = np.frombuffer(block, dtype=dtype, count=num_vertices)

    names = arr.dtype.names
    if not all(c in names for c in ("x", "y", "z")):
        raise ValueError("PLY vertex element lacks x/y/z.")

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1)
        # Open3D stores colors in [0, 1]; get_points_and_colors scales back to uint8.
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    return pcd


# Load a point cloud from local disk, or from Azure Blob when absent locally.
def load_o3d_pcd(file_path: str):
    if has_blob_config() and should_use_blob(file_path):
        data = read_blob_media_bytes(file_path)
        return _parse_binary_ply_bytes(data)
    return o3d.io.read_point_cloud(file_path)


# Get points and colors from a Open3D point cloud
def get_points_and_colors(pcd: o3d.geometry.PointCloud):
    points = np.asarray(pcd.points)
    colors = np.zeros_like(points, dtype=np.uint8)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        if colors.max() < 1.1:
            colors = (colors * 255).astype(np.uint8)
    return points, colors


# Preprocess a point cloud
def cleanup_pcd(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    num_nb: int = 15,
    std_ratio: float = 2.0,
):
    # voxelize the point cloud
    pcd = pcd.voxel_down_sample(voxel_size)
    # remove outliers
    pcd, _ = pcd.remove_statistical_outlier(num_nb, std_ratio=std_ratio)
    return pcd
