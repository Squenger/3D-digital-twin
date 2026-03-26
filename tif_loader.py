"""
tif_loader.py
---------------
Module responsible for reading, sorting, and normalizing TIF images
from a given directory to prepare them for processing.
"""

import re
from pathlib import Path

import numpy as np
import tifffile


class TifLoader:
    """
    Class to load and chronologically (or numerically) order
    TIF image slices present in a directory.
    """

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Directory not found: {self.folder}")

    def load(self, fast_test: bool = False) -> list[np.ndarray]:
        """
        Scans the directory, identifies image files (.tif or .tiff),
        sorts them in a logical (natural) order, and returns a list of NumPy arrays.
        """
        files = sorted(
            [f for f in self.folder.iterdir() if f.suffix.lower() in {".tif", ".tiff"} and not f.name.startswith('.')],
            key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)],
        )
        if not files:
            raise FileNotFoundError(f"No TIF files found in: {self.folder}")

        if fast_test and len(files) > 200:
            # Keep only the first 500 images BEFORE reading them from disk
            files = files[0:500]
            print("[Loader] FAST TEST ENABLED: Loading limited to the first 500 images.")

        print(f"[Loader] Reading {len(files)} TIF images...")
        slices = []
        for i, f in enumerate(files, 1):
            slices.append(self._read(f))
            if i % 100 == 0 or i == len(files):
                print(f"  > {i}/{len(files)} images read.")
                
        return slices

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        """
        Reads a specific TIF file and normalizes it to 8-bit (grayscale).
        Automatically handles multi-page images, RGB format, and intensity scaling.
        """
        img = tifffile.imread(str(path))
        
        # Handling multi-page files (keeping only the first page)
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = img[0]
            
        # Converting RGB format to standard grayscale
        if img.ndim == 3:
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
            
        # Normalizing pixel intensity to an 8-bit range [0, 255]
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
            
        return img
