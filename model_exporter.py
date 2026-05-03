"""
Model export and visualization utilities.
"""

import logging
from pathlib import Path

import trimesh

logger = logging.getLogger(__name__)

try:
    import Mesh
    import Part
    FREECAD_AVAILABLE = True
except ImportError:
    FREECAD_AVAILABLE = False

try:
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Builder, TopoDS_Shell
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.gp import gp_Pnt
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False


class ModelExporter:
    """
    Expert system for 3D topological export.
    Supports surface formats (.stl, .obj, .ply, .glb) as well as 
    boundary representation (B-Rep) solid formats (.step) utilizing
    available CAD kernels computationally.
    """

    FORMATS = {".stl", ".obj", ".ply", ".glb", ".step"}

    def export(self, mesh: trimesh.Trimesh, filepath: str | Path, solid: bool = False) -> Path:
        """
        Saves the object's mesh to disk. The target file format 
        is automatically identified from the provided path extension.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The input triangulated surface mesh.
        filepath : str | Path
            The explicit disk destination for the structural model.
        solid : bool, optional
            If True, generates a rigorous mathematical B-Rep model appropriate for CAD,
            otherwise generates a polygonal mesh format. Default is False.

        Returns
        -------
        Path
            The unified saved path location.
            
        Raises
        ------
        ValueError
            If the requested extension format is unsupported.
        ImportError
            If volumetric export is requested but neither FreeCAD nor OCC libraries are present.
        IOError
            If pythonocc-core encounters an internal topology writing error.
        """
        filepath = Path(filepath).resolve()
        ext = filepath.suffix.lower()
        if ext not in self.FORMATS:
            raise ValueError(f"Format '{ext}' not supported. Use: {', '.join(sorted(self.FORMATS))}")

        logger.info(f"Starting backup to: {filepath.name}...")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not solid:
            if ext == ".glb":
                logger.info("GLB format detected. Creating the scene.")
                data = trimesh.scene.Scene(geometry={"model": mesh}).export(file_type="glb")
            else:
                logger.info(f"Format {ext} detected. Export in progress (may take time depending on size).")
                data = mesh.export(file_type=ext.lstrip("."))

            if isinstance(data, str):
                data = data.encode()
            
            logger.info("Writing physical data to disk.")
            filepath.write_bytes(data)
        else:
            if not FREECAD_AVAILABLE and not OCC_AVAILABLE:
                raise ImportError("FreeCAD (Mesh/Part) or pythonocc-core (OCC) is required for solid/STEP export.\n"
                                "Please install one of them (e.g., 'conda install -c conda-forge freecad' or 'pythonocc-core').")
            
            logger.info(f"Volumetric mathematical export ({ext}) in progress...")
            
            if FREECAD_AVAILABLE:
                # --- Path A: FreeCAD (Preferred Method) ---
                logger.info("Using FreeCAD CAD geometric kernel...")
                temp_stl = filepath.with_suffix(".temp.stl")
                mesh.export(str(temp_stl))
                try:
                    fc_mesh = Mesh.read(str(temp_stl))
                    shape = Part.Shape()
                    shape.makeShapeFromMesh(fc_mesh.Topology, 0.1)
                    fc_solid = Part.makeSolid(shape)
                    refined_solid = fc_solid.removeSplitter()
                    if ext == ".step" or ext == ".stp":
                        refined_solid.exportStep(str(filepath))
                    else:
                        Part.export([refined_solid], str(filepath))
                finally:
                    if temp_stl.exists(): temp_stl.unlink()
            
            elif OCC_AVAILABLE:
                # --- Path B: pythonocc-core (Fallback Method) ---
                logger.info("Using OpenCASCADE (pythonocc) CAD topological kernel as fallback...")
                writer = STEPControl_Writer()
                Interface_Static.SetCVal("write.step.schema", "AP203")
                
                # We need to build a shape from triangles
                # For performance, we'll sew them
                sewing = BRepBuilderAPI_Sewing(1e-3) # 0.001mm tolerance
                
                v = mesh.vertices
                f = mesh.faces
                
                logger.info(f"Converting {len(f)} spatial triangles to Boundary Representations (B-Rep)...")
                for face in f:
                    p1 = gp_Pnt(float(v[face[0]][0]), float(v[face[0]][1]), float(v[face[0]][2]))
                    p2 = gp_Pnt(float(v[face[1]][0]), float(v[face[1]][1]), float(v[face[1]][2]))
                    p3 = gp_Pnt(float(v[face[2]][0]), float(v[face[2]][1]), float(v[face[2]][2]))
                    
                    try:
                        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
                        poly = BRepBuilderAPI_MakePolygon(p1, p2, p3, True)
                        face_shape = BRepBuilderAPI_MakeFace(poly.Wire())
                        sewing.Add(face_shape.Face())
                    except Exception:
                        continue
                
                logger.info("Sewing computational faces into a continuous manifold shell...")
                sewing.Perform()
                sewn_shape = sewing.SewedShape()
                
                final_shape = sewn_shape
                try:
                    make_solid = BRepBuilderAPI_MakeSolid(final_shape)
                    if make_solid.IsDone():
                        final_shape = make_solid.Solid()
                except Exception:
                    logger.warning("Could not create an enclosed dense solid. Exporting as open exterior shell.")
                
                writer.Transfer(final_shape, STEPControl_AsIs)
                status = writer.Write(str(filepath))
                if status != 1:
                    raise IOError(f"Failed to write STEP file via OpenCASCADE (Status: {status})")

            logger.info(f"CAD programmatic export physically successful: {filepath.name}")

        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"File {filepath.name} successfully committed to storage ({size_mb:.1f} MB).")
        return filepath

    @staticmethod
    def visualize(mesh: trimesh.Trimesh) -> None:
        """
        Spawns an interactive 3D graphical viewport.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The triangulated data object to render.
        """
        logger.info("Spawning interactive 3D graphical window.")
        mesh.show()
