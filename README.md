# 3D Digital Twin: Real-Time Topography Prediction

## Project Objective
The primary objective of this project is to enable the real-time prediction and reconstruction of a 3D digital twin based on layer-by-layer topographical data gathered directly during the 3D printing process.

Traditional Micro-CT (Computed Tomography) scans provide high-fidelity 3D volume reconstructions and precise internal porosity calculations. However, these scans are extremely slow, expensive, and computatio nally intensive, making them completely unviable for fast iterations or real-time monitoring. 

To overcome this limitation, this project develops a Neural Network trained using high-quality CT scan reconstructions as the "ground truth". The ultimate goal is to process the fast, layer-by-layer topographical scans of an object currently being printed, and predict an instant reconstruction whose quality is virtually identical to a full CT scan—enabling real-time structural validation and 3D digital twin generation without the prohibitive time costs.

---

## Core Pipeline Orchestration (`main.py`)

The `main.py` script serves as the central operational engine for the 3D reconstruction and dataset preparation pipeline. It takes raw physical image slices and transforms them into mathematically sound volumetric bodies and measurable meshes.

### Key Functionalities

* **Automated Data Loading (`TifLoader`)**
  Reads, normalizes (to 8-bit grayscale), and chronologically sorts directories containing sequential `.tif`/`.tiff` image slices representing the physical scanned layers. Includes a "fast test" mode to artificially limit loading capacity for quick debugging.

* **Binarization & Morphological Cleaning (`SliceProcessor`)**
  Converts the raw images into clean binary maps separating printed matter from empty space (using automatic Multi-Otsu thresholding). It dynamically applies a bounding crop across the X, Y, and Z axes using standard-deviation limits to trim empty ambient space, drastically reducing memory overhead.

* **3D Volumetric Mesh Construction (`ModelBuilder`)**
  Translates the stacked 2D binary masks into an immense, contiguous 3D voxel grid. It utilizes the Marching Cubes algorithm to generate a precise 3D surface mesh natively representing the printed object. It includes noise-filtering mechanisms that discard disconnected phantom topologies below a certain face count.

* **Interactive Manual Geometry Extraction**
  Offers an interactive feedback loop (via GUI integration) allowing operators to explicitly limit the spatial domains and surgically isolate the relevant structural architecture before finalizing the 3D mesh.

* **Advanced Porosity Analysis (`VolumeAnalyzer`)**
  Conducts sophisticated internal void metrology:
  - Calculates the exact global porosity percentage within the solid matter.
  - Generates an explicit "negative" 3D mesh that visually models the internal pores.
  - Extracts 2D cross-sectional images on defined planes.
  - Creates analytical variance/porosity graphs charting structural drift per layer.

* **Export & Geometric Visualization (`ModelExporter`)**
  Provides a robust interface to save the final reconstructed topology into established 3D formats (e.g., standard `.stl` arrays or solid volumetric `.step` files for direct CAD integration) alongside a built-in viewer for immediate visual evaluation.

* **Scalable Batch Processing**
  Designed to execute the entire reconstruction and analysis pipeline entirely headless across vast arrays of experimental sub-directories, allowing high-throughput data extraction required for Neural Network generation limits.
