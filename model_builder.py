"""
model_builder.py
----------------------
Module responsible for converting 2D slices into a 3D volume (voxels),
and extracting the latter as a surface or solid mesh.
"""

import numpy as np
import trimesh
from skimage import measure


class ModelBuilder:
    """
    Class managing 3D geometric modeling.
    Allows for the generation of a mesh.
    """

    def __init__(self, layer_thickness: float = 0.1, voxel_size: float = 0.1, component_faces_threshold: int = 1000) -> None:
        # Real physical dimensions in millimeters to respect proportionality
        self.layer_thickness = layer_thickness  # Vertical resolution (Z axis) between each slice
        self.voxel_size = voxel_size            # Horizontal and lateral resolution (X or Y axes)
        self.component_faces_threshold = component_faces_threshold  # Minimum number of faces to keep a piece
        self.voxels: np.ndarray | None = None

    def build(self, slices: list[np.ndarray], solid: bool = False) -> trimesh.Trimesh:
        """
        Orchestrates the complete conversion of successive 2D slices to a 3D mesh object,
        with the possibility of generating a dense (solid) or surface model.
        """
        print(f"[Builder] Stacking {len(slices)} slices into a 3D volume...")
        # Stacking 2D matrices to create the basic 3D volume (Z, Height, Width)
        self.voxels = np.stack(slices, axis=0).astype(np.uint8)
        z, h, w = self.voxels.shape
        print(f"  > Volume created: {w}x{h} px by {z} layers.")

        if solid:
            # Dense approach: each detected voxel is converted into a cube of matter
            print("[Builder] Generating solid model with voxels... (this may take a long time)")
            grid_bool = (self.voxels > 0)
            mesh = trimesh.voxel.VoxelGrid(grid_bool).as_boxes()
            
            # Dimensional adjustment (applying physical scale)
            # trimesh maps the NumPy 3D array orientation (Z, Y, X) to X, Y, Z representation
            mesh.apply_scale((self.voxel_size, self.voxel_size, self.layer_thickness))
        else:
            # Surface approach: extracting the contour (hull) using Marching Cubes
            # Applying padding (artificial margin) ensuring a properly closed mesh
            print("[Builder] Extracting the surface (this may take a long time)")
            grid = np.pad(self.voxels, pad_width=1, mode="constant")
    
            vertices, faces, normals, _ = measure.marching_cubes(
                grid, level=0.5,
                spacing=(self.layer_thickness, self.voxel_size, self.voxel_size),
            )
            print("[Builder] Finalizing the geometry")
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)

        if self.component_faces_threshold > 0:
            print(f"[Builder] Cleaning connected components (removing pieces < {self.component_faces_threshold} faces)...")
            try:
                components = mesh.split(only_watertight=False)
                if isinstance(components, list) and len(components) > 1:
                    valid_components = [c for c in components if len(c.faces) >= self.component_faces_threshold]
                    
                    if not valid_components:
                        # If the threshold was too demanding, we keep at least the largest piece
                        valid_components = [max(components, key=lambda c: len(c.faces))]
                        
                    mesh = trimesh.util.concatenate(valid_components)
                    print(f"  > Elements kept: {len(valid_components)} out of an initial total of {len(components)} components.")
            except Exception as e:
                print(f"  [Error] Unable to clean the mesh: {e}")

        print(f"  > Final 3D mesh generated ({len(mesh.faces)} faces).")
        return mesh
