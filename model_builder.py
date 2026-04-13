"""
model_builder.py
----------------------
Module responsible for converting 2D slices into a 3D volume (voxels),
and extracting the latter as a surface or solid mesh.
"""

import logging

import numpy as np
import trimesh
from skimage import measure

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Class managing 3D geometric modeling.
    Extracts a surface mesh from a discrete voxel grid using the Marching Cubes algorithm.
    """

    def __init__(self, layer_thickness: float = 0.1, voxel_size: float = 0.1, component_faces_threshold: int = 1000) -> None:
        """
        Initializes the ModelBuilder.

        Parameters
        ----------
        layer_thickness : float
            Real physical vertical resolution (Z axis) between each slice in mm.
        voxel_size : float
            Real physical horizontal and lateral resolution (X or Y axes) in mm.
        component_faces_threshold : int
            Minimum number of faces to keep a continuous floating piece (removes noise).
        """
        self.layer_thickness = layer_thickness
        self.voxel_size = voxel_size
        self.component_faces_threshold = component_faces_threshold
        self.voxels: np.ndarray | None = None

    def build(self, slices: list[np.ndarray], solid: bool = False) -> trimesh.Trimesh:
        """
        Orchestrates the complete conversion of successive 2D slices to a 3D mesh object,
        with the possibility of generating a dense (solid) or surface model.

        Parameters
        ----------
        slices : list of np.ndarray
            The sequence of 2D binary images.
        solid : bool, optional
            A flag indicating if the model is destined for solid representation.

        Returns
        -------
        mesh : trimesh.Trimesh
            The 3D reconstructed triangulated mesh.
        """
        logger.info(f"Stacking {len(slices)} slices into a 3D volume...")
        self.voxels = np.stack(slices, axis=0).astype(np.uint8)
        z, h, w = self.voxels.shape
        logger.debug(f"Volume created: {w}x{h} px by {z} layers.")

        logger.info("Extracting the surface (this may take a long time)...")
        grid = np.pad(self.voxels, pad_width=1, mode="constant")
    
        vertices, faces, normals, _ = measure.marching_cubes(
            grid, level=0.5,
            spacing=(self.layer_thickness, self.voxel_size, self.voxel_size),
        )
        logger.info("Finalizing the geometry...")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)

        if self.component_faces_threshold > 0:
            logger.info(f"Cleaning connected components (removing pieces < {self.component_faces_threshold} faces)...")
            try:
                components = mesh.split(only_watertight=False)
                if isinstance(components, list) and len(components) > 1:
                    valid_components = [c for c in components if len(c.faces) >= self.component_faces_threshold]
                    
                    if not valid_components:
                        valid_components = [max(components, key=lambda c: len(c.faces))]
                        
                    mesh = trimesh.util.concatenate(valid_components)
                    logger.debug(f"Elements kept: {len(valid_components)} out of an initial total of {len(components)} components.")
            except Exception as e:
                logger.error(f"Unable to clean the mesh: {e}")

        logger.info(f"Final 3D mesh generated ({len(mesh.faces)} faces).")
        return mesh
