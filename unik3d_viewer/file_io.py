import os
import numpy as np
import trimesh
import traceback

def save_glb(filepath, points, colors=None):
    """Saves a point cloud to a GLB file using trimesh."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        num_points = points.shape[0]

        point_colors = None
        if colors is not None and colors.shape[0] == num_points:
            colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
            point_colors = np.hstack((colors_uint8, np.full((num_points, 1), 255, dtype=np.uint8)))
        elif colors is not None:
            print(f"Warning: Color data shape mismatch or invalid for {filepath}. Saving without colors.")

        cloud = trimesh.points.PointCloud(vertices=points, colors=point_colors)
        cloud.export(filepath, file_type='glb')
    except Exception as e:
        print(f"Error saving GLB file {filepath}: {e}")
        traceback.print_exc()

def load_glb(filepath):
    """Loads points and colors from a GLB file using trimesh."""
    try:
        mesh = trimesh.load(filepath, file_type='glb', process=False)

        if isinstance(mesh, trimesh.points.PointCloud):
            points = np.array(mesh.vertices, dtype=np.float32)
            colors = None
            if hasattr(mesh, 'colors') and mesh.colors is not None and len(mesh.colors) == len(points):
                colors = np.array(mesh.colors[:, :3], dtype=np.float32) / 255.0
            return points, colors
        elif isinstance(mesh, trimesh.Trimesh):
             print(f"Warning: Loaded GLB {filepath} as Trimesh, using vertices only.")
             points = np.array(mesh.vertices, dtype=np.float32)
             colors = None
             if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) == len(points):
                 colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
             return points, colors
        else:
            print(f"Warning: Loaded GLB {filepath} is not a PointCloud or Trimesh.")
            return None, None
    except Exception as e:
        print(f"Error loading GLB file {filepath}: {e}")
        return None, None 