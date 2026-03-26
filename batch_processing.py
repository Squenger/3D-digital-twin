import argparse
import os
from pathlib import Path
from main import TifTo3D

def batch_process():
    print("[Batch Process] Starting batch processing of TIF samples to 3D")
    
    parent_folder = input("Please enter the path of the folder containing the sample folders: ").strip()
    target_folder = input("Please enter the path of the target folder for output (.stl) files: ").strip()

    parent_path = Path(parent_folder)
    target_path = Path(target_folder)

    if not parent_path.exists() or not parent_path.is_dir():
        print(f"[Error] The source folder '{parent_path}' does not exist or is not a directory.")
        return

    # Create the target folder if it does not exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Retrieve all subdirectories (samples)
    samples = [d for d in parent_path.iterdir() if d.is_dir()]

    if not samples:
        print(f"[Error] No subdirectories (samples) found in '{parent_path}'.")
        return

    print(f"\n[Batch Process] {len(samples)} samples found. Starting batch processing.")

    # Default parameters from main.py
    threshold = None
    layer_thickness = 0.015
    voxel_size = 0.015
    visualize = False
    solid = False
    fast_test = False
    component_threshold = 10000

    for i, sample_dir in enumerate(samples, 1):
        sample_name = sample_dir.name
        output_path = target_path / f"{sample_name}.stl"
        
        print(f"\n[Batch Process] [{i}/{len(samples)}] Processing sample: {sample_name}")
        print(f"  > Input folder: {sample_dir}")
        print(f"  > Output file: {output_path}")
        
        try:
            pipeline = TifTo3D(
                input_folder=sample_dir,
                output_path=output_path,
                threshold=threshold,
                layer_thickness=layer_thickness,
                voxel_size=voxel_size,
                visualize=visualize,
                solid=solid,
                fast_test=fast_test,
                component_threshold=component_threshold,
            )
            pipeline.execute()
            print(f"  > Model successfully saved.")
        except Exception as e:
            print(f"  [Error] Failed processing sample {sample_name}: {e}")

    print("\n[Batch Process] Batch processing finished.")

if __name__ == "__main__":
    batch_process()
