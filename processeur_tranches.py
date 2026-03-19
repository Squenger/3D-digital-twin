"""
processeur_tranches.py
----------------------
Module de traitement d'image dédié à la binarisation et au filtrage morphologique
des tranches individuelles avant leur assemblage 3D.
"""

import cv2
import numpy as np


class ProcesseurTranches:
    """
    Classe effectuant le prétraitement des images 2D (tranches).
    Elle gère la binarisation.
    """

    def __init__(self, seuil: int | None = None) -> None:
        # choix de la méthode de seuillage :
        # - None : algorithme d'Otsu (calcul automatique et optimal du seuil)
        # - int : application d'une valeur de seuil stricte imposée ([0, 255])
        self.seuil = seuil

    def traiter_lot(self, tranches: list[np.ndarray]) -> list[np.ndarray]:
        """
        Applique la binarisation sur l'ensemble de la pile d'images en entrée.
        """
        total = len(tranches)
        print(f"Binarisation de {total} images...")
        resultat = []
        for i, t in enumerate(tranches, start=1):
            resultat.append(self._traiter(t))
            if i % 100 == 0 or i == total:
                print(f"  > {i}/{total} images binarisées.")
        return resultat

    def _traiter(self, img: np.ndarray) -> np.ndarray:
        """
        Passe la tranche en phase de binarisation (séparation matière/arrière-plan).
        """
        # Binarisation automatique avec Otsu ou par seuil absolu fixe
        flag = cv2.THRESH_BINARY + (cv2.THRESH_OTSU if self.seuil is None else 0)
        seuil_val = 0 if self.seuil is None else self.seuil
        _, bin_img = cv2.threshold(img, seuil_val, 255, flag)

        return (bin_img > 0).astype(np.uint8)
