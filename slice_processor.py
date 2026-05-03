"""
Binarization and morphological filtering of slices.
"""

import logging

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

logger = logging.getLogger(__name__)


class SliceProcessor:
    """
    Class that performs the preprocessing of 2D images (slices).
    It manages the binarization and automatic bounding volume extraction.
    
    Attributes
    ----------
    threshold : int | None
        Manual binarization threshold. If None, Multi-Otsu is used.
    window_size : int
        Size of the central region used to compute standard deviations (in pixels).
    std_dev_z : float
        Standard deviation threshold to determine empty boundaries along the Z axis.
    std_dev_y : float
        Standard deviation threshold to determine empty boundaries along the Y axis.
    std_dev_x : float
        Standard deviation threshold to determine empty boundaries along the X axis.
    """

    def __init__(self, threshold: int | None = None, window_size: int = 400,
                 std_dev_z: float = 28.0, std_dev_y: float = 28.0, std_dev_x: float = 28.0) -> None:
        self.threshold = threshold
        self.window_size = window_size
        self.std_dev_z = std_dev_z
        self.std_dev_y = std_dev_y
        self.std_dev_x = std_dev_x

    def _determine_bounds(self, slices: list[np.ndarray]) -> tuple[int, int, int, int, int, int]:
        """ 
        Identifies the boundaries (Z, Y, X) of the sample based on center uniformity. 
        """
        if not slices:
            return 0, 0, 0, 0, 0, 0
            
        h, w = slices[0].shape
        depth = len(slices)
        
        # Center zone coordinates for variance detection
        wy = self.window_size // 2
        wx = self.window_size // 2
        
        y_mid, x_mid = h // 2, w // 2
        y1_c, y2_c = max(0, y_mid - wy), min(h, y_mid + wy)
        x1_c, x2_c = max(0, x_mid - wx), min(w, x_mid + wx)
        
        z_start, z_end = 0, depth - 1
        # 1. Search for Z start: std in center window drops at empty slices
        for i in range(depth):
            if np.std(slices[i][y1_c:y2_c, x1_c:x2_c]) > self.std_dev_z:
                z_start = i
                break
                
        # 2. Search for Z end (bottom to top)
        for i in range(depth - 1, -1, -1):
            if np.std(slices[i][y1_c:y2_c, x1_c:x2_c]) > self.std_dev_z:
                z_end = i
                break
                
        # For Y and X, analyze a block of valid Z slices in the middle
        z_mid = (z_start + z_end) // 2 if z_end >= z_start else depth // 2
        z_margin = min(20, max(1, (z_end - z_start) // 10))
        z1_c = max(z_start, z_mid - z_margin)
        z2_c = min(z_end, z_mid + z_margin)
        
        vol_3d = np.stack(slices[z1_c:z2_c+1], axis=0).astype(np.float32)  # [Z, H, W]
        center_img = vol_3d.mean(axis=0)

        step = max(1, self.window_size // 20)

        # 3. Search for Y start / end
        y_start, y_end = 0, h - 1
        for i in range(0, h, step):
            if np.std(vol_3d[:, i:i+step, x1_c:x2_c]) > self.std_dev_z:
                y_start = max(0, i - step)
                break
        for i in range(h - 1, -1, -step):
            if np.std(vol_3d[:, max(0, i-step):i, x1_c:x2_c]) > self.std_dev_z:
                y_end = min(h - 1, i + step)
                break

        # 4. Search for X start / end
        x_start, x_end = 0, w - 1
        y_mid_valid = (y_start + y_end) // 2
        y1_c_val = max(0, y_mid_valid - wy)
        y2_c_val = min(h, y_mid_valid + wy)

        for i in range(0, w, step):
            if np.std(vol_3d[:, y1_c_val:y2_c_val, i:i+step]) > self.std_dev_z:
                x_start = max(0, i - step)
                break
        for i in range(w - 1, -1, -step):
            if np.std(vol_3d[:, y1_c_val:y2_c_val, max(0, i-step):i]) > self.std_dev_z:
                x_end = min(w - 1, i + step)
                break

        return z_start, z_end, y_start, y_end, x_start, x_end

    def process_batch(self, slices: list[np.ndarray], progress_callback=None, skip_auto_crop: bool = False) -> list[np.ndarray]:
        """
        Applies binarization to the entire input image stack.
        If skip_auto_crop is True, auto std_dev cropping is bypassed (used when manual crop is active).
        """
        if not slices:
            return []
            
        if skip_auto_crop:
            logger.info("Manual crop mode: skipping auto std_dev bounding. Processing all slices.")
            valid_slices = slices
        else:
            # Get bounds via uniformity (center standard deviation)
            z1, z2, y1, y2, x1, x2 = self._determine_bounds(slices)
            logger.info(f"Extracting 3D volume: Z[{z1}:{z2}], Y[{y1}:{y2}], X[{x1}:{x2}] (out of {len(slices)} slices).")
            # Crop the image list in 3D
            valid_slices = [s[y1:y2, x1:x2] for s in slices[z1:z2 + 1]]
        
        total = len(valid_slices)
        logger.info(f"Binarizing {total} images...")
        result = []

        # 3. Individual binarization of each isolated slice to avoid gaps
        for i, t in enumerate(valid_slices, start=1):
            t_bin, _ = self._process(t)
            result.append(t_bin)
            
            if i % 100 == 0 or i == total:
                logger.debug(f"{i}/{total} images binarized.")
                if progress_callback:
                    # Map the progress from 20% to 70%
                    pct = 20 + 50 * (i / total)
                    progress_callback(pct, 100, f"Binarization ({i}/{total})...")
                
        return result

    def _process(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Passes the slice through binarization phase (material/background separation).
        """
        threshold1 = threshold_multiotsu(img, classes=3)

        img = img.astype(np.uint8)

        img[img <= threshold1[1]] = 0 
        
        return (img > 0).astype(np.uint8), threshold1[1]
