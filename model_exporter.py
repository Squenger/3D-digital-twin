"""
model_exporter.py
---------------------
Module managing the export and interactive visualization of generated 3D models.
"""

from pathlib import Path

import trimesh


class ModelExporter:
    """
    Export to various standard 3D formats (.stl, .obj, .ply, .glb)
    and interactive display for visual validation.
    """

    FORMATS = {".stl", ".obj", ".ply", ".glb"}

    def export(self, mesh: trimesh.Trimesh, filepath: str | Path) -> Path:
        """
        Saves the object's mesh to disk. The target file format 
        is automatically identified from the provided path extension.
        """
        filepath = Path(filepath).resolve()
        ext = filepath.suffix.lower()
        if ext not in self.FORMATS:
            raise ValueError(f"Format '{ext}' not supported. Use: {', '.join(sorted(self.FORMATS))}")

        print(f"\n[Exporter] Starting backup to: {filepath.name}...")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Specific processing for the .glb format (requires encapsulation in a Scene)
        # For other formats (STL, OBJ, etc.), the export is done directly on the mesh
        if ext == ".glb":
            print("  > GLB format detected. Creating the scene.")
            data = trimesh.scene.Scene(geometry={"model": mesh}).export(file_type="glb")
        else:
            print(f"  > Format {ext} detected. Export in progress (may take time depending on size).")
            data = mesh.export(file_type=ext.lstrip("."))

        if isinstance(data, str):
            data = data.encode()
            
        print("  > Writing to disk.")
        filepath.write_bytes(data)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"[Exporter] File {filepath.name} successfully saved ({size_mb:.1f} MB).")
        return filepath

    @staticmethod
    def visualize(mesh: trimesh.Trimesh) -> None:
        """
        Opens an interactive 3D interface.
        """
        print("\n[Exporter] Opening the interactive 3D window.")
        mesh.show()
