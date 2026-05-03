# 3D Digital Twin: Topography-based Prediction

## Overview
This project focuses on real-time 3D reconstruction and prediction using topographical data collected during 3D printing. 

While Micro-CT scans provide high-quality volume data, they are too slow for real-time monitoring. This pipeline uses CT data as ground truth to train models that can reconstruct 3D volumes from faster layer-by-layer topography scans.

---

## Main Pipeline (`main.py`)
The `main.py` script manages the data processing from raw images to 3D meshes.

### Features
* **Data Loading (`TifLoader`):** Loads and sorts `.tif` slices. Includes a fast-debug mode.
* **Processing (`SliceProcessor`):** Binarization (Otsu) and auto-cropping to reduce memory usage.
* **3D Construction (`ModelBuilder`):** Voxel assembly and Marching Cubes mesh generation with noise filtering.
* **Manual Adjustment:** Interactive GUI for manual cropping and domain isolation.
* **Porosity Analysis (`VolumeAnalyzer`):** Calculates porosity, generates internal void meshes, and plots profiles per layer.
* **Export (`ModelExporter`):** Saves to STL or STEP formats and includes a basic 3D viewer.
* **Batch Mode:** Process entire directories of samples at once.
