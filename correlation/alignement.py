import numpy as np
import open3d as o3d
from skimage import measure, filters
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Pour le tracé 3D Matplotlib

class RecalageInverseEtVisualisationML:
    def __init__(self, ct_file, ply_dir, voxel_size, voxel_downsample_ransac, resolution_ply_xy=None, distance_z_couches=0.4):
        self.ct_file = ct_file
        self.ply_dir = ply_dir
        self.vx, self.vy, self.vz = voxel_size[2], voxel_size[1], voxel_size[0] # X, Y, Z
        self.voxel_downsample_ransac = voxel_downsample_ransac
        self.resolution_ply_xy = resolution_ply_xy
        self.distance_z_couches = distance_z_couches
        
        self.pcd_gocator_brut = None
        self.ct_tensor_brut = None
        self.binary_ct_pour_icp = None
        self.pcd_ct_surface = None
        self.matrice_icp = None # Matrice CT -> Gocator
        
        # Nouvelles dimensions fixées par le Gocator
        self.nouvelle_shape_zyx = None
        self.min_bound_physique = None
        
        # Données de sortie
        self.tenseur_gocator_input_binaire = None
        self.tenseur_ct_ground_truth_gris = None

    def preparer_donnees_initiales(self):
        import os
        import glob
        
        print("[1/5] Chargement et préparation des données...")
        
        # 1. Gocator : Chargement couche par couche
        fichiers_couches = glob.glob(os.path.join(self.ply_dir, "couche_nuage_*.ply"))
        # Trier par index de couche
        fichiers_couches.sort(key=lambda f: int(os.path.basename(f).replace('couche_nuage_', '').replace('.ply', '')))
        
        tous_les_points = []
        facteur_echelle_xy = 1.0
        
        if len(fichiers_couches) == 0:
            raise FileNotFoundError(f"Aucun fichier 'couche_nuage_*.ply' trouvé dans {self.ply_dir}")

        for i, fichier in enumerate(fichiers_couches):
            pcd_couche = o3d.io.read_point_cloud(fichier)
            points = np.asarray(pcd_couche.points)
            
            # Calcul du facteur d'échelle XY sur la première couche (si demandé)
            if self.resolution_ply_xy is not None and i == 0:
                unique_x = np.unique(np.round(points[:, 0], decimals=4))
                if len(unique_x) > 1:
                    dx_actuel = np.median(np.diff(unique_x))
                    facteur_echelle_xy = self.resolution_ply_xy / dx_actuel
                    print(f"      -> Résolution actuelle du PLY (X) : {dx_actuel:.5f} mm")
                    print(f"      -> Mise à l'échelle XY par le facteur : {facteur_echelle_xy:.5f} pour atteindre {self.resolution_ply_xy} mm")
                else:
                    print("      -> AVERTISSEMENT : Impossible de calculer dx_actuel sur la couche 0.")

            # Mise à l'échelle X et Y
            points[:, 0] *= facteur_echelle_xy
            points[:, 1] *= facteur_echelle_xy
            
            # Application de la distance Z selon le numéro de la couche
            # On applique un signe négatif pour reproduire l'inversion d'axe Z précédente
            points[:, 2] = - (i * self.distance_z_couches)
            
            tous_les_points.append(points)

        # Fusion de toutes les couches en un seul nuage
        points_gocator_complets = np.vstack(tous_les_points)
        self.pcd_gocator_brut = o3d.geometry.PointCloud()
        self.pcd_gocator_brut.points = o3d.utility.Vector3dVector(points_gocator_complets)
        
        print(f"      -> Nuage reconstruit : {len(fichiers_couches)} couches, {len(points_gocator_complets)} points totaux.")

        # Calcul de la Bounding Box du Gocator pour définir la future grille 3D
        self.min_bound_physique = self.pcd_gocator_brut.get_min_bound()
        max_bound = self.pcd_gocator_brut.get_max_bound()
        
        # Combien de voxels faut-il pour contenir le Gocator ?
        dim_x = int(np.ceil((max_bound[0] - self.min_bound_physique[0]) / self.vx))
        dim_y = int(np.ceil((max_bound[1] - self.min_bound_physique[1]) / self.vy))
        dim_z = int(np.ceil((max_bound[2] - self.min_bound_physique[2]) / self.vz))
        self.nouvelle_shape_zyx = (dim_z, dim_y, dim_x) # Numpy : Z, Y, X

        # 2. CT (Extraction de surface pour l'ICP uniquement)
        ct_data = np.load(self.ct_file)
        self.ct_tensor_brut = ct_data[ct_data.files[0]]
        
        
        # Échange des axes X et Y supprimé
        # print("      -> Échange des axes X et Y du CT...")
        # self.ct_tensor_brut = np.swapaxes(self.ct_tensor_brut, 1, 2)


        threshold = filters.threshold_otsu(self.ct_tensor_brut)
        self.binary_ct_pour_icp = self.ct_tensor_brut > threshold
        verts, faces, _, _ = measure.marching_cubes(self.binary_ct_pour_icp, level=0.5)

        # Mise à l'échelle (Voxels -> Millimètres physiques)
        verts[:, 0] *= self.vz # Z
        verts[:, 1] *= self.vy # Y
        verts[:, 2] *= self.vx # X

        mesh_ct = o3d.geometry.TriangleMesh()
        mesh_ct.vertices = o3d.utility.Vector3dVector(verts)
        mesh_ct.triangles = o3d.utility.Vector3iVector(faces)
        self.pcd_ct_surface = mesh_ct.sample_points_uniformly(number_of_points=1000000)

    def executer_recalage_inverse(self):
        print("[2/5] Calcul de la transformation spatiale inversée...")
        # ATTENTION : Source = Surface du CT, Target = Nuage Gocator Immobile
        source_down = self.pcd_ct_surface.voxel_down_sample(self.voxel_downsample_ransac)
        target_down = self.pcd_gocator_brut.voxel_down_sample(self.voxel_downsample_ransac)

        # Normales
        for pcd in [source_down, target_down, self.pcd_ct_surface, self.pcd_gocator_brut]:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_downsample_ransac*2, max_nn=30))

        # Descripteurs FPFH
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_downsample_ransac*5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_downsample_ransac*5, max_nn=100))

        # RANSAC
        distance_ransac = self.voxel_downsample_ransac * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, distance_ransac,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_ransac)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

        # ICP Point-to-Plane : Calcule la matrice pour déplacer la surface du CT vers le Gocator
        distance_icp = self.voxel_downsample_ransac * 0.4
        result_icp = o3d.pipelines.registration.registration_icp(
            self.pcd_ct_surface, self.pcd_gocator_brut, distance_icp, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

        self.matrice_icp = result_icp.transformation
        print(f"      -> Alignement (Fitness) : {result_icp.fitness * 100:.2f}%")
        print(f"      -> Erreur RMSE : {result_icp.inlier_rmse:.4f} mm")

    def generer_datasets_ml_alignes(self):
        print("[3/5] Génération des tenseurs d'entraînement parfaitement superposés...")
        
        # 1. INPUT ML : Voxelisation du Gocator Brut dans SA propre grille (Immobile)
        print("      -> Voxelisation du Gocator (Tenseur Input Binaire)...")
        points_goc = np.asarray(self.pcd_gocator_brut.points)
        idx_x = np.round((points_goc[:, 0] - self.min_bound_physique[0]) / self.vx).astype(int)
        idx_y = np.round((points_goc[:, 1] - self.min_bound_physique[1]) / self.vy).astype(int)
        idx_z = np.round((points_goc[:, 2] - self.min_bound_physique[2]) / self.vz).astype(int)

        self.tenseur_gocator_input_binaire = np.zeros(self.nouvelle_shape_zyx, dtype=np.uint8)
        
        mask_securite = (idx_z >= 0) & (idx_z < self.nouvelle_shape_zyx[0]) & \
                        (idx_y >= 0) & (idx_y < self.nouvelle_shape_zyx[1]) & \
                        (idx_x >= 0) & (idx_x < self.nouvelle_shape_zyx[2])
        self.tenseur_gocator_input_binaire[idx_z[mask_securite], idx_y[mask_securite], idx_x[mask_securite]] = 1

        # 2. GROUND TRUTH ML : Transformation Affine du volume Micro-CT complet
        print("      -> Transformation du volume Micro-CT complet (Interpolation 3D Cubique)...")
        
        # Construction de la matrice pour scipy.ndimage (Nécessite l'inverse de la matrice ICP)
        T_inv = np.linalg.inv(self.matrice_icp)

        # Composition des transformations : New_Idx -> Goc_Phys -> CT_Phys -> Old_Idx
        M_total = np.diag([1/self.vx, 1/self.vy, 1/self.vz, 1]) @ T_inv @ np.diag([self.vx, self.vy, self.vz, 1])
        M_total[0:3, 3] = T_inv[0:3, 3] # Translation physique reste physique

        # order=3 : Interpolation cubique pour préserver les niveaux de gris
        self.tenseur_ct_ground_truth_gris = ndi.affine_transform(
            self.ct_tensor_brut, 
            matrix=M_total[0:3, 0:3], 
            offset=M_total[0:3, 3], 
            output_shape=self.nouvelle_shape_zyx, 
            order=3, 
            cval=0.0 # Remplissage de noir à l'extérieur
        )

    def visualiser_3d_sous_echantillonne(self, facteur_sous_echantillon):
        print(f"[4/5] Génération de la visualisation 3D améliorée (Facteur={facteur_sous_echantillon})...")
        
        # 1. Extraction des points du Tenseur Gocator Binaire
        coords_goc = np.argwhere(self.tenseur_gocator_input_binaire == 1)
        points_goc = coords_goc

        # 2. Extraction des points du Tenseur CT Ground Truth (Binarisé pour le tracé)
        print("      -> Binarisation du CT interpolé pour extraction des points...")
        threshold_ml = filters.threshold_otsu(self.tenseur_ct_ground_truth_gris)
        coords_ct = np.argwhere(self.tenseur_ct_ground_truth_gris > threshold_ml)
        points_ct = coords_ct[::facteur_sous_echantillon]

        # Tracé Matplotlib
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Vérification 3D de la Superposition (Points extraits des Tenseurs)")

        # Gocator en Rouge
        ax.scatter(points_goc[:, 2], points_goc[:, 1], points_goc[:, 0], c='red', s=1, label='Input Gocator Binaire')
        # CT en Bleu
        ax.scatter(points_ct[:, 2], points_ct[:, 1], points_ct[:, 0], c='blue', s=1, label='Ground Truth CT (Binarisé)')

        ax.set_xlabel('X (Voxels)')
        ax.set_ylabel('Y (Voxels)')
        ax.set_zlabel('Z (Voxels)')
        ax.legend()
        
        # Vue axonométrique pour mieux apprécier la superposition
        ax.view_init(elev=30, azim=45) 
        plt.tight_layout()
        plt.show()

    def verifier_2d_et_sauvegarder(self, out_input, out_gt, out_img, save_npz=True):
        print("[5/5] Génération de la figure 2D et sauvegarde des datasets ML...")
        if save_npz:
            print(f"      -> Sauvegarde des fichiers NPZ : {out_input}, {out_gt}")
            np.savez_compressed(out_input, self.tenseur_gocator_input_binaire)
            np.savez_compressed(out_gt, self.tenseur_ct_ground_truth_gris)
        else:
            print("      -> Sauvegarde des fichiers NPZ désactivée.")

        milieu_z = self.nouvelle_shape_zyx[0] // 2
        threshold_ml = filters.threshold_otsu(self.tenseur_ct_ground_truth_gris)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"INPUT (Gocator Voxelisé Z={milieu_z})")
        plt.imshow(self.tenseur_gocator_input_binaire[milieu_z, :, :], cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title(f"SUPERPOSITION (Gocator Rouge sur CT Gris Z={milieu_z})")
        # Fond : CT Ground Truth (Niveaux de gris)
        plt.imshow(self.tenseur_ct_ground_truth_gris[milieu_z, :, :], cmap='gray', alpha=1.0)
        # Premier plan : Gocator Input (Masque rouge semi-transparent)
        plt.imshow(self.tenseur_gocator_input_binaire[milieu_z, :, :], cmap='Reds', alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_img)
        print("Figure 2D sauvegardée. Tenseurs sauvegardés. Prêt pour l'entraînement !")

def main(params):
    pipeline = RecalageInverseEtVisualisationML(
        ct_file=params['FICHIER_CT'],
        ply_dir=params['DOSSIER_PLY'],
        voxel_size=params['VOXEL_SIZE'],
        voxel_downsample_ransac=params['VOXEL_DOWNSAMPLE_RANSAC'],
        resolution_ply_xy=params.get('RESOLUTION_PLY_XY', None),
        distance_z_couches=params.get('DISTANCE_Z_COUCHES', 0.4)
    )
    pipeline.preparer_donnees_initiales()
    pipeline.executer_recalage_inverse()
    pipeline.generer_datasets_ml_alignes()
    
    # Nouvelle visualisation 3D interactive
    pipeline.visualiser_3d_sous_echantillonne(params['VISU_3D_SOUS_ECHANTILLON'])
    
    # Vérification 2D statique et sauvegarde
    pipeline.verifier_2d_et_sauvegarder(
        params['OUT_INPUT_NPZ'], 
        params['OUT_GT_NPZ'], 
        params['OUT_IMAGE_2D'],
        save_npz=params.get('SAVE_NPZ', True)
    )

if __name__ == "__main__":
    # =========================================================================
    # PARAMÈTRES UTILISATEUR (À MODIFIER ICI)
    # =========================================================================
    config = {
        'FICHIER_CT': "correlation/MICRO CT/outputs/tensor_S5_15um_Original_post_traitement.npz",
        'DOSSIER_PLY': "sortie/GOCATOR/5",
        
        # Dimensions physiques RÉELLES d'un voxel CT (en mm, ordre Z, Y, X)
        'VOXEL_SIZE': (0.015, 0.015, 0.015), 
        
        # Résolution du Gocator en X et Y (si None, utilise la résolution du fichier)
        'RESOLUTION_PLY_XY': 0.05,
        
        # Distance entre les couches en Z pour la reconstruction du PLY
        'DISTANCE_Z_COUCHES': 0.4,
        
        # Sous-échantillonnage pour le calcul RANSAC (plus petit = plus précis/lent)
        'VOXEL_DOWNSAMPLE_RANSAC': 0.05,
        
        # Paramètre de visualisation 3D : Afficher 1 voxel sur X
        # Augmenter ce chiffre (ex: 100) si le tracé 3D est trop lent
        'VISU_3D_SOUS_ECHANTILLON': 250,
        
        # Sorties pour le Deep Learning
        'OUT_INPUT_NPZ': "ML_Input_Gocator.npz", 
        'OUT_GT_NPZ': "ML_GroundTruth_CT.npz",     
        'OUT_IMAGE_2D': "verif_ml.png",
        
        # Options
        'SAVE_NPZ': False  # Mettre à False pour ne pas sauvegarder les fichiers .npz
    }
    main(config)