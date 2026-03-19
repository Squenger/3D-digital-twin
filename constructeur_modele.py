"""
constructeur_modele.py
----------------------
Module responsable de la conversion des tranches 2D en un volume 3D (voxels),
et de l'extraction de ce dernier sous la forme d'un maillage surfacique ou plein.
"""

import numpy as np
import trimesh
from skimage import measure


class ConstructeurModele:
    """
    Classe gérant la modélisation 3D géométrique. 
    Permet la génération d'un maillage.
    """

    def __init__(self, epaisseur_couche: float = 0.1, taille_voxel: float = 0.1) -> None:
        # Dimensions physiques réelles en millimètres pour respecter la proportionnalité
        self.epaisseur_couche = epaisseur_couche  # Résolution verticale (axe Z) entre chaque tranche
        self.taille_voxel = taille_voxel          # Résolution horizontale et latérale (axes X ou Y)
        self.voxels: np.ndarray | None = None

    def construire(self, tranches: list[np.ndarray], plein: bool = False) -> trimesh.Trimesh:
        """
        Orchestre la conversion complète des tranches 2D successives vers un objet maillage 3D,
        avec la possibilité de générer un modèle dense (plein) ou surfacique.
        """
        print(f"Empilement de {len(tranches)} tranches en volume 3D...")
        # Empilement des matrices 2D pour créer le volume de base 3D (Z, Hauteur, Largeur)
        self.voxels = np.stack(tranches, axis=0).astype(np.uint8)
        z, h, w = self.voxels.shape
        print(f"  > Volume créé : {w}x{h} px par {z} couches.")

        if plein:
            # Approche dense : chaque voxel détecté est converti en un cube de matière
            print("Génération du modèle plein avec voxels... (peut être très long)")
            grille_bool = (self.voxels > 0)
            maillage = trimesh.voxel.VoxelGrid(grille_bool).as_boxes()
            
            # Ajustement dimensionnel (application de l'échelle physique)
            # trimesh mappe l'orientation du tableau 3D NumPy (Z, Y, X) vers la représentation X, Y, Z
            maillage.apply_scale((self.taille_voxel, self.taille_voxel, self.epaisseur_couche))
        else:
            # Approche surfacique : extraction du contour (coque) utilisant les Marching Cubes
            # Application d'un padding (marge artificielle) garantissant un maillage correctement fermé
            print("Extraction de la surface via Marching Cubes... ( long)")
            grille = np.pad(self.voxels, pad_width=1, mode="constant")
    
            sommets, faces, normales, _ = measure.marching_cubes(
                grille, level=0.5,
                spacing=(self.epaisseur_couche, self.taille_voxel, self.taille_voxel),
            )
            print("Finalisation de la géométrie...")
            maillage = trimesh.Trimesh(vertices=sommets, faces=faces, vertex_normals=normales, process=True)

        print(f"  > Maillage 3D généré ({len(maillage.faces)} faces).")
        return maillage
