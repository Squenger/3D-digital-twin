"""
main.py
-------
Main script to run the 3D reconstruction pipeline.
Loads images, binarizes, builds 3D models, and handles exports.
"""

import argparse
import logging
import sys
from pathlib import Path

from model_builder import ModelBuilder
from model_exporter import ModelExporter
from post_processing import VolumeAnalyzer
from slice_processor import SliceProcessor
from tif_loader import TifLoader

logger = logging.getLogger(__name__)


class TifTo3D:
    """
    Main class to manage the 3D reconstruction process.
    
    Steps:
        1. Load data (TifLoader)
        2. Preprocess/Binarize (SliceProcessor)
        3. Build 3D Mesh (ModelBuilder)
        4. Export/View (ModelExporter)
        5. Analyze Porosity (VolumeAnalyzer)
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
        progress_callback=None,
        interactive_crop_callback=None,
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

        # Submodule Initializations
        self.loader = TifLoader(self.input_folder)
        self.processor = SliceProcessor(threshold=threshold, window_size=window_size,
                                         std_dev_z=std_dev_z, std_dev_y=std_dev_y, std_dev_x=std_dev_x)
        self.builder = ModelBuilder(layer_thickness, voxel_size, component_threshold)
        self.exporter = ModelExporter()

    def execute(self) -> None:
        """Runs the pipeline."""
        if self.progress_callback: self.progress_callback(0, 100, "Loading raw imaging data...")
        slices = self.loader.load(fast_test=self.fast_test, progress_callback=self.progress_callback)
        
        # Always run the standard pipeline first (auto-crop Z/Y/X + binarization)
        # This ensures clean binarization regardless of whether manual crop is used.
        if self.progress_callback: self.progress_callback(20, 100, "Binarization and morphological cleaning...")
        skip_crop = self.interactive_crop_callback is not None
        bin_slices = self.processor.process_batch(slices, progress_callback=self.progress_callback, skip_auto_crop=skip_crop)
        
        if self.progress_callback: self.progress_callback(70, 100, "Constructing the 3D topographical mesh...")
        mesh = self.builder.build(bin_slices, solid=self.solid)
        
        # Manual crop is a secondary refinement step on the already clean binarized voxels.
        if self.interactive_crop_callback and self.builder.voxels is not None:
            if self.progress_callback: self.progress_callback(80, 100, "Waiting for manual geometry extraction...")
            manual_bounds = self.interactive_crop_callback(self.builder.voxels)
            if manual_bounds:
                z1, z2, y1, y2, x1, x2 = manual_bounds
                logger.info(f"Applying manual post-reconstruction crop limit: Z[{z1}:{z2}], Y[{y1}:{y2}], X[{x1}:{x2}]")
                self.builder.voxels = self.builder.voxels[z1:z2+1, y1:y2+1, x1:x2+1]
                
                if self.progress_callback: self.progress_callback(85, 100, "Rebuilding the cropped 3D architectural mesh...")
                mesh = self.builder.build(list(self.builder.voxels), solid=self.solid)
        
        if self.output_path:
            if self.progress_callback: self.progress_callback(90, 100, "Encoding and saving the structural model...")
            self.exporter.export(mesh, self.output_path, solid=self.solid)
            
        if self.post_process and self.builder.voxels is not None:
            analyzer = VolumeAnalyzer(self.builder.voxels)
            
            # 1. Exact internal porosity
            porosity = analyzer.estimate_porosity()
            self.porosity_rate = porosity
            logger.info(f"EXACT INTERNAL POROSITY: {porosity:.2f} %")

            # 2. Optionally generate the 3D Porosity Model
            if self.generate_porosity_mesh:
                if self.progress_callback: self.progress_callback(95, 100, "Generating porosity theoretical 3D model...")
                holes_3d = analyzer.get_internal_porosity_voxels()
                porosity_mesh = self.builder.build(list(holes_3d), solid=self.solid)
                
                if self.output_path:
                    p_out = self.output_path.with_name(f"{self.output_path.stem}_porosity.stl")
                    self.exporter.export(porosity_mesh, p_out, solid=self.solid)
                    self.porosity_model_path = p_out
                    logger.info(f"Porosity mathematical model saved to {p_out}")

            # 3. Cross sections
            export_folder = self.output_path.parent if self.output_path else self.input_folder
            model_name = self.output_path.stem if self.output_path else "model_preview"
            
            self.generated_slices, self.slice_resolutions = analyzer.export_cross_sections(
                output_folder=export_folder, 
                model_name=model_name,
                num_slices=150,
                save_to_disk=self.save_slices,
                voxel_size=self.builder.voxel_size,
                layer_thickness=self.builder.layer_thickness
            )
            
            graph_path = export_folder / f"{model_name}_porosity_profile.png"
            saved_graph = analyzer.plot_porosity_profiles(save_path=graph_path)
            if saved_graph:
                self.generated_graphs.append(saved_graph)
            
        if self.visualize:
            self.exporter.visualize(mesh)
            
        if self.progress_callback: self.progress_callback(100, 100, "Done.")


def configure_logging(verbose: bool) -> None:
    """Configures the standard Python logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout
    )

def main() -> None:
    # Take every argument from the command line and run execute with it
    parser = argparse.ArgumentParser(description="3D Digital Twin Volume Reconstruction Pipeline")
    parser.add_argument("-i", "--input", type=str, default='/Volumes/KINGSTON/DATA_MICRO_CT/sample 3/Sample 3 15um/S3_15um_Original', help="Input directory containing .tif slices (or parent folder if batch mode).")
    parser.add_argument("-o", "--output", type=str, default='./sortie/modele3(20).stl', help="Output file path for the 3D model (or target directory if batch mode).")
    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode on multiple subdirectories.")
    parser.add_argument("--solid", action="store_true", help="Generate a solid volumetric CAD model (.step).")
    parser.add_argument("--fast-test", action="store_true", help="Fast debug mode limited to the first 500 images.")
    parser.add_argument("--visualize", action="store_true", help="Open the interactive 3D viewer at completion.")
    parser.add_argument("--no-post-process", action="store_true", help="Disable porosity analysis and slice extraction.")
    parser.add_argument("--save-slices", action="store_true", help="Save cross sections physically to disk instead of memory.")
    parser.add_argument("--porosity-mesh", action="store_true", help="Generate an explicit 3D mesh of the object's pores.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
    
    # Mathematical hyper-parameters
    parser.add_argument("--layer-thickness", type=float, default=0.015, help="Physical distance between two successive slices in mm.")
    parser.add_argument("--voxel-size", type=float, default=0.015, help="Physical X/Y resolution in mm.")
    parser.add_argument("--component-threshold", type=int, default=100000, help="Faces limit below which structural components are discarded.")
    parser.add_argument("--window-size", type=int, default=400, help="Uniformity analysis crop window size.")
    parser.add_argument("--std-z", type=float, default=28.0, help="Variance limit for Z axis.")
    parser.add_argument("--std-y", type=float, default=28.0, help="Variance limit for Y axis.")
    parser.add_argument("--std-x", type=float, default=28.0, help="Variance limit for X axis.")
    
    args = parser.parse_args()
    configure_logging(args.verbose)

    try:
        if args.batch:
            # Batch Execution Mode
            input_dir = Path(args.input)
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            samples = [d for d in input_dir.iterdir() if d.is_dir()]
            if not samples:
                logger.error(f"No sample subdirectories found inside batch folder '{input_dir}'")
                return
                
            logger.info(f"Targeting {len(samples)} samples for batch parallel reconstruction.")
            for sample in samples:
                target_file = output_dir / f"{sample.name}.step" if args.solid else output_dir / f"{sample.name}.stl"
                logger.info(f"--- Processing Batch Unit: {sample.name} ---")
                
                pipeline = TifTo3D(
                    input_folder=sample,
                    output_path=target_file,
                    threshold=None,
                    layer_thickness=args.layer_thickness,
                    voxel_size=args.voxel_size,
                    visualize=False, # Disable visualization in batch to prevent halting
                    solid=args.solid,
                    fast_test=args.fast_test,
                    component_threshold=args.component_threshold,
                    post_process=not args.no_post_process,
                    save_slices=args.save_slices,
                    generate_porosity_mesh=args.porosity_mesh,
                    window_size=args.window_size,
                    std_dev_z=args.std_z,
                    std_dev_y=args.std_y,
                    std_dev_x=args.std_x,
                )
                pipeline.execute()
        else:
            # Single Sample Execution Mode
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            pipeline = TifTo3D(
                input_folder=args.input,
                output_path=out_path,
                threshold=None,
                layer_thickness=args.layer_thickness,
                voxel_size=args.voxel_size,
                visualize=args.visualize,
                solid=args.solid,
                fast_test=args.fast_test,
                component_threshold=args.component_threshold,
                post_process=not args.no_post_process,
                save_slices=args.save_slices,
                generate_porosity_mesh=args.porosity_mesh,
                window_size=args.window_size,
                std_dev_z=args.std_z,
                std_dev_y=args.std_y,
                std_dev_x=args.std_x,
            )
            pipeline.execute()
        
    except Exception as e:
        logger.error(f"Critical execution failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
