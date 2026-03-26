"""
main.py
--------------
Main script.
It handles loading TIF images, binarization, 3D reconstruction,
and exporting the final model.
"""

import time
from pathlib import Path

from tif_loader import TifLoader
from model_builder import ModelBuilder
from model_exporter import ModelExporter
from slice_processor import SliceProcessor
from post_processing import VolumeAnalyzer


class TifTo3D:
    """
    Class managing 3D reconstruction.
    
    Architecture of steps:
        1. Loading raw data (TifLoader)
        2. Preprocessing and binarization (SliceProcessor)
        3. Volume assembly (Mesh) (ModelBuilder)
        4. Save or visualization (ModelExporter)
    """

    def __init__(
        self,
        input_folder: str | Path,
        output_path: str | Path | None = None,
        threshold: int | None = None,
        layer_thickness: float = 0.1,
        voxel_size: float = 0.1,
        visualize: bool = False,
        solid: bool = False,
        fast_test: bool = False,
        component_threshold: int = 10000,
        post_process: bool = True,
        generate_porosity_mesh: bool = False,
        save_slices: bool = True,
        window_size: int = 400,
        std_dev_z: float = 28.0,
        std_dev_y: float = 21.0,
        std_dev_x: float = 21.0,
        progress_callback = None,
        interactive_crop_callback = None,
    ) -> None:
        self.input_folder = Path(input_folder)
        self.output_path = Path(output_path) if output_path else None
        self.visualize = visualize
        self.solid = solid
        self.fast_test = fast_test
        self.post_process = post_process
        self.generate_porosity_mesh = generate_porosity_mesh
        self.save_slices = save_slices
        self.progress_callback = progress_callback
        self.interactive_crop_callback = interactive_crop_callback
        
        self.generated_slices = []
        self.generated_graphs = []

        # Initialization of submodules
        self.loader     = TifLoader(self.input_folder)
        self.processor  = SliceProcessor(threshold=threshold, window_size=window_size,
                                         std_dev_z=std_dev_z, std_dev_y=std_dev_y, std_dev_x=std_dev_x)
        self.builder    = ModelBuilder(layer_thickness, voxel_size, component_threshold)
        self.exporter   = ModelExporter()

    def execute(self) -> None:
        if self.progress_callback: self.progress_callback(0, 100, "Loading images...")
        slices = self.loader.load(fast_test=self.fast_test)
        
        # Always run the standard pipeline first (auto-crop Z/Y/X + binarization)
        # This ensures clean binarization regardless of whether manual crop is used.
        if self.progress_callback: self.progress_callback(20, 100, "Binarization and cleaning...")
        skip_crop = self.interactive_crop_callback is not None
        bin_slices = self.processor.process_batch(slices, progress_callback=self.progress_callback, skip_auto_crop=skip_crop)
        
        if self.progress_callback: self.progress_callback(70, 100, "Building the 3D mesh...")
        mesh = self.builder.build(bin_slices, solid=self.solid)
        
        # Manual crop is a secondary refinement step on the already clean binarized voxels.
        # This preserves binarization quality and model size in both modes.
        if self.interactive_crop_callback and self.builder.voxels is not None:
            if self.progress_callback: self.progress_callback(80, 100, "Waiting for manual crop...")
            manual_bounds = self.interactive_crop_callback(self.builder.voxels)
            if manual_bounds:
                z1, z2, y1, y2, x1, x2 = manual_bounds
                print(f"[Processor] Applying manual post-reconstruction crop: Z[{z1}:{z2}], Y[{y1}:{y2}], X[{x1}:{x2}]")
                self.builder.voxels = self.builder.voxels[z1:z2+1, y1:y2+1, x1:x2+1]
                
                if self.progress_callback: self.progress_callback(85, 100, "Rebuilding the cropped 3D mesh...")
                mesh = self.builder.build(list(self.builder.voxels), solid=self.solid)
        
        if self.output_path:
            if self.progress_callback: self.progress_callback(90, 100, "Saving the model...")
            self.exporter.export(mesh, self.output_path)
            
        if self.post_process and self.builder.voxels is not None:
            analyzer = VolumeAnalyzer(self.builder.voxels)
            
            # 1. Exact internal porosity
            porosity = analyzer.estimate_porosity()
            self.porosity_rate = porosity
            print(f"[Post-Processing] EXACT INTERNAL POROSITY: {porosity:.2f} %")

            # 2. Optionally generate the 3D Porosity Model
            if self.generate_porosity_mesh:
                if self.progress_callback: self.progress_callback(95, 100, "Generating porosity 3D model...")
                holes_3d = analyzer.get_internal_porosity_voxels()
                porosity_mesh = self.builder.build(list(holes_3d), solid=self.solid)
                
                if self.output_path:
                    p_out = self.output_path.with_name(f"{self.output_path.stem}_porosity.stl")
                    self.exporter.export(porosity_mesh, p_out)
                    self.porosity_model_path = p_out
                    print(f"[Post-Processing] Porosity 3D model saved to {p_out}")

            # 3. Cross sections
            export_folder = self.output_path.parent if self.output_path else self.input_folder
            model_name = self.output_path.stem if self.output_path else "model_preview"
            
            self.generated_slices = analyzer.export_cross_sections(
                output_folder=export_folder, 
                model_name=model_name,
                num_slices=150, # Increased slice count
                save_to_disk=self.save_slices
            )
            
            graph_path = export_folder / f"{model_name}_porosity_profile.png"
            saved_graph = analyzer.plot_porosity_profiles(save_path=graph_path)
            if saved_graph:
                self.generated_graphs.append(saved_graph)
            
        if self.visualize:
            self.exporter.visualize(mesh)
            
        if self.progress_callback: self.progress_callback(100, 100, "Complete.")



def main() -> None:

    
    # Inputs / Outputs 
    input_folder  = '/Volumes/KINGSTON/DATA_MICRO_CT/sample 3/Sample 3 15um/S3_15um_Original'
    output_path   = './sortie/modele3(20).stl'  # Set to None to disable disk export
    
    # Rendering and Post-processing 
    visualize   = False  # Displays the interactive 3D viewer after export
    solid       = False  # Generates a solid volumetric model (True) or a surface model (False)
    fast_test   = True   # (DEBUG OPTION) Test only on the first 500 slices 
    post_process = True  # Activates porosity estimation and slice export
    save_slices  = False  # Save subsets to disk vs just memory GUI display
    
    # Algorithmic and physical parameters 
    threshold           = None  # Manual binarization threshold [0-255]. None = automatic (Otsu)
    layer_thickness     = 0.015  # Distance between two successive slices in mm)
    voxel_size          = 0.015  # X Y resolution (mm)
    component_threshold = 100000 # Any isolated 3D piece with less than X faces will be removed
    window_size         = 400    # Crop window size at the center of the image to check for standard deviation
    std_dev_z           = 28.0   # Standard deviation threshold for Z (vertical)
    std_dev_y           = 28.0   # Standard deviation threshold for Y (frontal)
    std_dev_x           = 28.0   # Standard deviation threshold for X (sagittal)

    try:
        pipeline = TifTo3D(
            input_folder=input_folder,
            output_path=output_path,
            threshold=threshold,
            layer_thickness=layer_thickness,
            voxel_size=voxel_size,
            visualize=visualize,
            solid=solid,
            fast_test=fast_test,
            component_threshold=component_threshold,
            post_process=post_process,
            save_slices=save_slices,
            window_size=window_size,
            std_dev_z=std_dev_z,
            std_dev_y=std_dev_y,
            std_dev_x=std_dev_x,
        )
        pipeline.execute()
        
    except Exception as e:
        print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
