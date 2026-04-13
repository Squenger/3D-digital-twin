"""
tif_loader.py
---------------
Module responsible for reading, sorting, and normalizing TIF images
from a given directory to prepare them for processing.
"""

import re
import logging
from pathlib import Path

import numpy as np
import tifffile


logger = logging.getLogger(__name__)


class TifLoader:
    """
    Class to load and chronologically (or numerically) order
    TIF image slices present in a directory.

    Attributes
    ----------
    folder : Path
        The specific folder containing the raw `.tif` or `.tiff` images.
    """

    def __init__(self, folder: str | Path) -> None:
        """
        Initializes the TifLoader.

        Parameters
        ----------
        folder : str or Path
            The directory path containing the sample TIF slices.
            
        Raises
        ------
        NotADirectoryError
            If the provided folder path does not exist or is not a directory.
        """
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {self.folder}")

    def load(self, fast_test: bool = False, progress_callback=None) -> list[np.ndarray]:
        """
        Scans the directory, identifies image files (.tif or .tiff),
        sorts them in a logical (natural) order, and returns a list of NumPy arrays.

        Parameters
        ----------
        fast_test : bool, optional
            If True, only loads the first 500 images for rapid debugging. Default is False.
        progress_callback : callable, optional
            Callback function to report progress.

        Returns
        -------
        slices : list of np.ndarray
            A chronologically ordered list of 2D image matrices.

        Raises
        ------
        FileNotFoundError
            If no valid TIF images are found in the directory.
        """
        files = sorted(
            [f for f in self.folder.iterdir() if f.suffix.lower() in {".tif", ".tiff"} and not f.name.startswith('.')],
            key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)],
        )
        if not files:
            raise FileNotFoundError(f"No TIF files found in: {self.folder}")

        if fast_test and len(files) > 200:
            files = files[0:500]
            logger.info("FAST TEST ENABLED: Loading limited to the first 500 images.")

        logger.info(f"Reading {len(files)} TIF images...")
        slices = []
        total = len(files)
        for i, f in enumerate(files, 1):
            slices.append(self._read(f))
            if progress_callback:
                pct = 20 * (i / total)
                progress_callback(pct, 100, f"Chargement des fichiers : {i}/{total} ({f.name})")
            
            if i % 100 == 0 or i == total:
                logger.debug(f"{i}/{total} images read.")
                
        return slices

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        """
        Reads a specific TIF file and normalizes it to 8-bit (grayscale).
        Automatically handles multi-page images, RGB format, and intensity scaling.

        Parameters
        ----------
        path : Path
            The explicit path to a single `.tif` image.

        Returns
        -------
        img : np.ndarray
            The normalized 8-bit (uint8) grayscale image array.
        """
        img = tifffile.imread(str(path))
        
        # Handle multi-page files (keeping only the first page)
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = img[0]
            
        # Convert RGB format to standard grayscale
        if img.ndim == 3:
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
            
        # Normalize pixel intensity to an 8-bit range [0, 255]
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
            
        return img
