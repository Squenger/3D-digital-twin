import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path

def main(params):
    data_dir = Path(params.get("DATA_DIR", "/Volumes/KINGSTON/DATA_MICRO_CT/sample 1/Sample 1 15um/S1_15um_Front_Cropped"))
    output_image = params.get("OUTPUT_IMAGE", "cross_sections.png")
    output_npz = params.get("OUTPUT_NPZ", "reconstructed_tensor.npz")
    save_npz = params.get("SAVE_NPZ", True)
    show_cross_sections = params.get("SHOW_CROSS_SECTIONS", True)
    show_3d_vis = params.get("SHOW_3D_VIS", True)
    downsample_3d_step = params.get("DOWNSAMPLE_3D_STEP", 4)
    max_points_3d = params.get("MAX_POINTS_3D", 100000)
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return
        
    print(f"Listing TIFF files from {data_dir}...")
    files = sorted(
        [f for f in data_dir.iterdir() if f.suffix.lower() in {".tif", ".tiff"} and not f.name.startswith('.')],
        key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)],
    )
    
    if not files:
        print(f"No TIFF files found in {data_dir}")
        return
        
    print(f"Found {len(files)} files. Loading...")
    slices = []
    total = len(files)
    
    import cv2
    for i, f in enumerate(files):
        try:
            img = tifffile.imread(str(f))
        except Exception as e:
            print(f"tifffile failed for {f.name}: {e}. Trying cv2...")
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"cv2 also failed for {f.name}. Skipping.")
                continue
                
        # Some cv2 reads might return 3D arrays for 2D images if not strictly grayscale
        if img.ndim > 2:
            img = img[:, :, 0] # Take first channel if it's unexpected
            
        slices.append(img)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Loaded {i + 1}/{total} images.")
            
    print("Stacking slices into a 3D tensor...")
    tensor = np.stack(slices)
    print(f"Tensor reconstructed successfully. Shape: {tensor.shape}, dtype: {tensor.dtype}")
    
    print("Isolating the main piece (removing background)...")
    from skimage.filters import threshold_otsu
    from scipy.ndimage import label as scilabel
    
    # Calculate threshold on downsampled tensor to save memory/time
    thresh = threshold_otsu(tensor[::4, ::4, ::4])
    print(f"Otsu Threshold calculated: {thresh}")
    
    # Create mask
    mask = tensor > thresh
    
    print("Labeling connected components to find the main piece...")
    labels, num_features = scilabel(mask)
    del mask # Free memory
    
    print(f"Found {num_features} components. Finding the largest one...")
    counts = np.bincount(labels.ravel())
    counts[0] = 0 # Ignore background (label 0)
    largest_label = np.argmax(counts)
    
    print("Applying mask to zero out everything outside the main piece...")
    piece_mask = (labels == largest_label)
    tensor[~piece_mask] = 0
    
    del labels
    del piece_mask
    print("Isolation complete.")
    
    if save_npz:
        npz_path = Path(__file__).resolve().parent / output_npz
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving tensor to {npz_path}...")
        np.savez_compressed(npz_path, tensor=tensor)
        print(f"Tensor saved successfully to {npz_path}.")

    if show_cross_sections:
        print("Generating verification image with multiple cross-sections...")
        z_max, y_max, x_max = tensor.shape
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle("3D Tensor Cross-Sections (Z, Y, X planes)", fontsize=20)
    
        # Good level scale (ignore 0.1% outliers on non-zero elements)
        non_zero = tensor[tensor > 0]
        if len(non_zero) > 0:
            vmin, vmax = np.percentile(non_zero, (0.1, 99.9))
        else:
            vmin, vmax = 0, 255
        print(f"Color scale vmin: {vmin}, vmax: {vmax}")
        
        # Z cross sections (XY planes)
        z_slices = [int(z_max * f) for f in [0.2, 0.4, 0.6, 0.8]]
        for i, z in enumerate(z_slices):
            ax = axes[0, i]
            ax.imshow(tensor[z, :, :], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"Z-Slice {z}")
            ax.axis('off')
            
        # Y cross sections (XZ planes)
        y_slices = [int(y_max * f) for f in [0.2, 0.4, 0.6, 0.8]]
        for i, y in enumerate(y_slices):
            ax = axes[1, i]
            ax.imshow(tensor[:, y, :], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"Y-Slice {y}")
            ax.axis('off')
            
        # X cross sections (YZ planes)
        x_slices = [int(x_max * f) for f in [0.2, 0.4, 0.6, 0.8]]
        for i, x in enumerate(x_slices):
            ax = axes[2, i]
            ax.imshow(tensor[:, :, x], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"X-Slice {x}")
            ax.axis('off')
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save in current directory
        output_path = Path(__file__).resolve().parent / output_image
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Verification image saved to {output_path}")

    if show_3d_vis:
        print("Generating lightweight 3D visualization...")
        # Downsample tensor to avoid memory/rendering freeze
        step = downsample_3d_step
        downsampled = tensor[::step, ::step, ::step]
        
        z_coords, y_coords, x_coords = np.nonzero(downsampled)
        vals = downsampled[z_coords, y_coords, x_coords]
        
        # Limit points for interactivity in Matplotlib
        max_points = max_points_3d
        if len(z_coords) > max_points:
            print(f"Sub-sampling from {len(z_coords)} points to {max_points} points for fluidity...")
            indices = np.random.choice(len(z_coords), max_points, replace=False)
            z_coords, y_coords, x_coords, vals = z_coords[indices], y_coords[indices], x_coords[indices], vals[indices]
            
        fig_3d = plt.figure(figsize=(10, 8))
        ax = fig_3d.add_subplot(111, projection='3d')
        
        # Scatter plot colored by intensity
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=vals, cmap='viridis', s=2, alpha=0.5)
        fig_3d.colorbar(scatter, ax=ax, label='Intensité (Niveaux de gris)')
        
        ax.set_title("Visualisation 3D (Échantillonnée)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        output_3d = Path(__file__).resolve().parent / "3d_visualization.png"
        plt.savefig(output_3d, dpi=150, bbox_inches='tight')
        print(f"3D Visualization saved to {output_3d}")

    print("Affichage des fenêtres interactives...")
    plt.show()

if __name__ == "__main__":
    params = {
        "DATA_DIR": "/Volumes/KINGSTON/DATA_MICRO_CT/sample 5/Sample 5 15um/S5_15um_Front_Cropped",
        "OUTPUT_IMAGE": "cross_sections5.png",
        "OUTPUT_NPZ": "reconstructed_tensor_sample5.npz",
        "SAVE_NPZ": True,
        "SHOW_3D_VIS": False,
        "DOWNSAMPLE_3D_STEP": 4,
        "MAX_POINTS_3D": 100000,
        "SHOW_CROSS_SECTIONS": True,
    }
    main(params)
