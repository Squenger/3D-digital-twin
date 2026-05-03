# 3D Digital Twin: Pipeline Architecture

This document describes the modules and data flow of the 3D reconstruction pipeline, from raw 2D images to 3D meshes and porosity analysis.

## Execution Flow
`TifLoader` -> `SliceProcessor` -> `ModelBuilder` -> `VolumeAnalyzer` -> `ModelExporter`

---

## 1. Data Ingestion: `TifLoader`
**File:** `tif_loader.py`

Scans a directory for TIFF files and loads them sequentially. Normalizes images to 8-bit grayscale to ensure consistent thresholding.

---

## 2. Processing: `SliceProcessor`
**File:** `slice_processor.py`

Handles binarization and cropping.
- **Auto-Cropping:** Detects sample boundaries based on pixel intensity variance to crop empty space.
- **Binarization:** Uses Otsu's method to separate matter from background.

---

## 3. Reconstruction: `ModelBuilder`
**File:** `model_builder.py`

Converts the voxel grid into a continuous mesh.
- **Marching Cubes:** Generates a polygonal surface around the binarized volume.
- **Filtering:** Removes small disconnected artifacts based on a face count threshold.

---

## 4. Analysis: `VolumeAnalyzer`
**File:** `post_processing.py`

Calculates internal voids and profiles.
- **Porosity:** Compares the actual volume with a morphologically filled version to estimate porosity.
- **Cross-sections:** Generates images along X, Y, and Z planes for inspection.

---

## 5. Export: `ModelExporter`
**File:** `model_exporter.py`

Saves the geometry to disk.
- **Meshes:** STL, OBJ, GLB.
- **Solids (STEP):** Optional conversion to B-Rep format using CAD libraries.

---

## 6. Execution Contexts
- **CLI (`main.py`):** Orchestrates the pipeline with support for batch processing.
- **GUI (`gui.py`):** Tkinter-based interface for interactive runs and manual cropping.
