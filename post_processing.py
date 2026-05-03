"""
Post-processing: cross-section extraction and porosity analysis.
"""

import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndimage
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class VolumeAnalyzer:
    """
    Class responsible for mathematical porosity analysis and 2D cross-section extraction.

    Attributes
    ----------
    voxels : np.ndarray
        The binarized 3D voxel volume (0s for background, 1s for matter).
    """

    def __init__(self, voxels: np.ndarray):
        """
        Initializes with the binarized voxel volume.
        """
        self.voxels = voxels

    def estimate_porosity(self) -> float:
        """
        Calculates the exact internal porosity of the volume.
        It uses morphological hole filling on each slice to find the true solid object
        without its internal pores, avoiding the bounding box empty spaces entirely.

        Returns
        -------
        porosity : float
            The internal porosity expressed as a percentage [0.0 - 100.0].
        """
        matter_volume = 0
        void_volume = 0
        
        for i in range(self.voxels.shape[0]):
            slice_2d = self.voxels[i, :, :]
            if slice_2d.max() > 0:
                filled = ndimage.binary_fill_holes(slice_2d)
                void_volume += (filled.astype(np.uint8) - slice_2d).sum()
                matter_volume += slice_2d.sum()
                
        total_volume = matter_volume + void_volume
        if total_volume == 0:
            return 0.0
        
        return (void_volume / total_volume) * 100.0
        
    def get_internal_porosity_voxels(self) -> np.ndarray:
        """
        Computes the internal pores explicitly.

        Returns
        -------
        holes_volume : np.ndarray
            A 3D matrix where internal holes are 1 and solid matter/exterior is 0.
        """
        holes_volume = np.zeros_like(self.voxels)
        for i in range(self.voxels.shape[0]):
            slice_2d = self.voxels[i, :, :]
            if slice_2d.max() > 0:
                filled = ndimage.binary_fill_holes(slice_2d)
                holes = filled.astype(np.uint8) - slice_2d
                holes_volume[i, :, :] = holes
        return holes_volume

    def plot_porosity_profiles(self, save_path: str | Path | None = None) -> Path | None:
        """
        Calculates and plots the internal porosity profile slice by slice across all 3 axes.

        Parameters
        ----------
        save_path : str | Path | None, optional
            Where to save the generated PNG plot. If None, it is not saved to disk.

        Returns
        -------
        Path | None
            The Path where the graph was saved, or None if no material was found.
        """
        axis_info = [(0, "Z (Vertical)"), (1, "Y (Frontal)"), (2, "X (Sagittal)")]
        
        fig = Figure(figsize=(18, 5))
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3)
        fig.suptitle("Internal Porosity Profile across all axes")
        
        # Calculate 3D internal holes from the Z-axis perspective (most reliable for cylinders)
        holes_3d = self.get_internal_porosity_voxels()
        
        # Find bounds to ignore the empty space in the profile
        y_v, x_v = np.where(self.voxels.sum(axis=0) > 0)
        z_v = np.where(self.voxels.sum(axis=(1, 2)) > 0)[0]
        
        if len(z_v) == 0:
            return None
            
        bounds = [
            (z_v.min(), z_v.max()),
            (y_v.min(), y_v.max()),
            (x_v.min(), x_v.max())
        ]
        
        for idx, (axis_id, axis_name) in enumerate(axis_info):
            porosities = []
            
            cmin, cmax = bounds[axis_id]
            for i in range(cmin, cmax + 1):
                if axis_id == 0:
                    voids = holes_3d[i, :, :].sum()
                    matter = self.voxels[i, :, :].sum()
                elif axis_id == 1:
                    voids = holes_3d[:, i, :].sum()
                    matter = self.voxels[:, i, :].sum()
                else:
                    voids = holes_3d[:, :, i].sum()
                    matter = self.voxels[:, :, i].sum()
                    
                total = voids + matter
                if total > 0:
                    porosities.append((voids / total) * 100.0)
                else:
                    porosities.append(0.0)
                    
            axes[idx].plot(range(cmin, cmax + 1), porosities, color='orange')
            axes[idx].set_title(f"Axis {axis_name}")
            axes[idx].set_xlabel("Slice Index")
            axes[idx].set_ylabel("Porosity (%)")
            axes[idx].grid(True)
            
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Porosity profile graphs saved to {save_path}")
            return Path(save_path)
        else:
            return None

    def export_cross_sections(
        self, 
        output_folder: str | Path, 
        model_name: str, 
        num_slices: int = 50, 
        save_to_disk: bool = True,
        voxel_size: float = 0.015,
        layer_thickness: float = 0.015
    ) -> tuple[dict[str, list[Path]], dict[str, tuple[float, float]]]:
        """
        Generates genuine 2D images of the slices within the model on XY and XZ planes (ignoring Z).

        Parameters
        ----------
        output_folder : str | Path
            Directory where the cross sections will be saved.
        model_name : str
            Prefix used in naming the exported directory.
        num_slices : int, optional
            Number of slices to generate per plane.
        save_to_disk : bool, optional
            Whether to write to the permanent user folder or to a temporary folder instead.
        voxel_size : float, optional
            Physical X/Y resolution in mm.
        layer_thickness : float, optional
            Physical Z resolution in mm.

        Returns
        -------
        tuple
            A tuple containing:
            - Dictionary mapping to a list of `Path` objects of generated slices.
            - Dictionary mapping the resolutions (x, y) for each axis.
        """
        if save_to_disk:
            path = Path(output_folder) / f"Cross_Sections_{model_name}"
        else:
            path = Path(tempfile.gettempdir()) / f"Cross_Sections_Temp_{model_name}"
            
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting cross sections to: {path}")
        
        axis_info = [(0, "Z_Axial"), (1, "Y_Frontal"), (2, "X_Sagittal")]
        saved_files = {"Z_Axial": [], "Y_Frontal": [], "X_Sagittal": []}
        
        # Get object bounding box to crop empty space around the slices
        y_v, x_v = np.where(self.voxels.sum(axis=0) > 0)
        z_v = np.where(self.voxels.sum(axis=(1, 2)) > 0)[0]
        
        if len(z_v) == 0:
            return saved_files
            
        z1, z2 = z_v.min(), z_v.max()
        y1, y2 = y_v.min(), y_v.max()
        x1, x2 = x_v.min(), x_v.max()
        
        for axis_id, axis_name in axis_info:
            sub_folder = path / axis_name
            sub_folder.mkdir(exist_ok=True)
            
            if axis_id == 0:
                depth = z2 - z1 + 1
            elif axis_id == 1:
                depth = y2 - y1 + 1
            else:
                depth = x2 - x1 + 1
            
            # Non-linear sampling: arcsin concentrates points near the center/edges
            t = np.linspace(-1, 1, num_slices)
            normalized_indices = np.arcsin(t) / np.pi + 0.5
            indices = np.unique((normalized_indices * (depth - 1)).astype(int))
            
            for offset in indices:
                if axis_id == 0:
                    i = z1 + offset
                    img = self.voxels[i, y1:y2+1, x1:x2+1]
                elif axis_id == 1:
                    i = y1 + offset
                    # Crop Z and X
                    img = self.voxels[z1:z2+1, i, x1:x2+1]
                else:
                    i = x1 + offset
                    # Crop Z and Y
                    img = self.voxels[z1:z2+1, y1:y2+1, i]
                    
                display_img = (img * 255).astype(np.uint8)
                fpath = sub_folder / f"slice_{axis_name}_{i:04d}.png"
                cv2.imwrite(str(fpath), display_img)
                saved_files[axis_name].append(fpath)
                
        total_files = sum(len(v) for v in saved_files.values())
        logger.info(f"{total_files} cross sections generated successfully.")
        
        resolutions = {
            "Z_Axial": (voxel_size, voxel_size),
            "Y_Frontal": (voxel_size, layer_thickness),
            "X_Sagittal": (voxel_size, layer_thickness)
        }
        return saved_files, resolutions
