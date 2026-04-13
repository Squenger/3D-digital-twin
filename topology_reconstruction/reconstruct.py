import re
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import Delaunay, cKDTree

def extract_layer_number(filename: str) -> int:
    matches = re.findall(r'\d+', filename)
    return int(matches[-1]) if matches else 0

def load_topology_layer(filepath: Path, layer_index: int, layer_thickness: float) -> np.ndarray:
    """
    Load points from a text file.
    Coordinate system: All units in millimeters (mm).
    Data format: [Z, X, Y] (Z is at index 0).
    """
    try:
        data = np.loadtxt(filepath)
    except Exception:
        return None
        
    # if data.size == 0 or data.ndim == 1:
    #     if data.size > 0 and data.ndim == 1:
    #         data = data.reshape(1, -1)
    #     else:
    #         return None
        
    # if data.shape[1] < 3:
    #     return None
        
    points = data.copy()[:,:3]/100
    
    z_median = np.median(points[:, 2])
    
    # 2. Filtrage : Supprime les points dont la distance à la Z médiane est > épaisseur / 2
    mask = np.abs(points[:, 2] - z_median) <= (layer_thickness / 2.0)
    points = points[mask]
    
    # median_z = (layer_index + 0.5) * layer_thickness
    # points[:, 2] += median_z
    # for i in points[:,0]:
    #     if i
    return points

def reconstruct_topology(input_folder: str, thickness: float, output_file: str = "nuage_de_points.ply"):
    """
    Reconstruct 3D surface and point cloud from sequential topology layers in mm.
    """
    input_path = Path(input_folder)
    if not input_path.exists() or not input_path.is_dir():
        return
        
    txt_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == '.txt' and not f.name.startswith('.')]
    if not txt_files:
        return
        
    txt_files_sorted = sorted(txt_files, key=lambda x: extract_layer_number(x.name))
        
    all_points = []
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer_out_dir = Path("sortie/GOCATOR/6")
    layer_out_dir.mkdir(parents=True, exist_ok=True)
    
    for filepath in txt_files_sorted:
        layer_num = extract_layer_number(filepath.name) - 1
        
        layer_points = load_topology_layer(filepath, layer_num, thickness)
        
        if layer_points is not None and len(layer_points) > 20:



            #  Statistical Outlier Removal (SOR)
            tree = cKDTree(layer_points)
            distances, _ = tree.query(layer_points, k=51)
            
            avg_distances = np.mean(distances[:, 1:], axis=1)
            
            mean_dist = np.mean(avg_distances)
            std_dist = np.std(avg_distances)
            
            threshold = mean_dist + 2.0 * std_dist
            
            valid_mask = avg_distances <= threshold
            layer_points = layer_points[valid_mask]
            
            all_points.append(layer_points)
            
            couche_nuage = trimesh.points.PointCloud(layer_points)
            couche_nuage.export(str(layer_out_dir / f"couche_nuage_{layer_num}.ply"))
            
            if len(layer_points) >= 3:
                tri = Delaunay(layer_points[:, 0:2])
                mesh = trimesh.Trimesh(vertices=layer_points, faces=tri.simplices, process=True)
                mesh.export(str(layer_out_dir / f"couche_surface_{layer_num}.stl"))
                
    if not all_points:
        return
            
    # Concatenate all layer 
    global_point_cloud = np.vstack(all_points)
    
    cloud = trimesh.points.PointCloud(global_point_cloud)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cloud.export(str(output_path))
    except Exception:
        pass


        


if __name__ == "__main__":
    INPUT_DIR = '/Volumes/My Book/data recov/Gocator_Data/ALIGNED/6'
    LAYER_THICKNESS = 0.3
    OUTPUT_PATH = '/Users/aiminemeddeb/Documents/3D TWIN/3D-digital-twin/sortie/GOCATOR/6/nuage_de_points2.ply'
    
    reconstruct_topology(
        input_folder=INPUT_DIR,
        thickness=LAYER_THICKNESS,
        output_file=OUTPUT_PATH)
