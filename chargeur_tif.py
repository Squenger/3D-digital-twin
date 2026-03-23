"""
chargeur_tif.py
---------------
Module responsable de la lecture, du tri et de la normalisation des images TIF
provenant d'un dossier donné, afin de préparer leur traitement.
"""

import re
from pathlib import Path

import numpy as np
import tifffile


class ChargeurTif:
    """
    Classe permettant de charger et d'ordonner chronologiquement (ou numériquement)
    les tranches d'images TIF présentes dans un répertoire.
    """

    def __init__(self, dossier: str | Path) -> None:
        self.dossier = Path(dossier)
        if not self.dossier.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {self.dossier}")

    def charger(self, test_rapide: bool = False) -> list[np.ndarray]:
        """
        Analyse le dossier, identifie les fichiers image (.tif ou .tiff),
        les trie de manière logique (naturelle) et retourne une liste de tableaux NumPy.
        """
        fichiers = sorted(
            [f for f in self.dossier.iterdir() if f.suffix.lower() in {".tif", ".tiff"} and not f.name.startswith('.')],
            key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)],
        )
        if not fichiers:
            raise FileNotFoundError(f"Aucun fichier TIF dans : {self.dossier}")

        print(f"[Chargeur] Lecture de {len(fichiers)} images TIF...")
        tranches = []
        for i, f in enumerate(fichiers, 1):
            tranches.append(self._lire(f))
            if i % 100 == 0 or i == len(fichiers):
                print(f"  > {i}/{len(fichiers)} images lues.")
                
        if test_rapide and len(tranches) > 200:
            milieu = len(tranches) // 2
            tranches = tranches[0: 500]
            print(f"\n[Chargeur] ⚡️ TEST RAPIDE : On ne garde que les 50 images du milieu (de l'indice {milieu - 25} à {milieu + 24}).")
                
        return tranches

    @staticmethod
    def _lire(chemin: Path) -> np.ndarray:
        """
        Lit un fichier TIF spécifique et le normalise sur 8 bits (niveaux de gris).
        Gère automatiquement les images multi-pages, le format RVB et l'échelle d'intensité.
        """
        img = tifffile.imread(str(chemin))
        
        # Gestion des fichiers multi-pages (conservation de la première page uniquement)
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = img[0]
            
        # Conversion du format RVB vers des niveaux de gris standards
        if img.ndim == 3:
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
            
        # Normalisation de l'intensité des pixels sur une plage de 8 bits [0, 255]
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
            
        return img
