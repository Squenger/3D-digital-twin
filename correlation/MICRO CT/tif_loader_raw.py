"""
tif_loader_raw.py
-----------------
Copie adaptée de tif_loader.py pour le module de corrélation.
Charge les images .tif en conservant le dtype d'origine (uint16, float32...)
SANS normalisation en uint8 ni binarisation.

Aucun fichier du projet original n'a été modifié.
"""

import re
import logging
from pathlib import Path

import numpy as np
import tifffile


logger = logging.getLogger(__name__)


class TifLoaderRaw:
    """
    Charge et trie les tranches .tif d'un dossier en conservant
    les valeurs d'intensité d'origine (pas de normalisation uint8).

    Attributes
    ----------
    folder : Path
        Dossier contenant les images .tif brutes.
    """

    def __init__(self, folder: str | Path) -> None:
        """
        Parameters
        ----------
        folder : str or Path
            Chemin du dossier contenant les tranches .tif.

        Raises
        ------
        NotADirectoryError
            Si le chemin n'existe pas ou n'est pas un dossier.
        """
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(f"Dossier introuvable : {self.folder}")

    def load(self, fast_test: bool = False) -> list[np.ndarray]:
        """
        Parcourt le dossier, trie les fichiers naturellement et retourne
        une liste de tableaux NumPy avec les intensités d'origine (float32).

        Parameters
        ----------
        fast_test : bool, optional
            Si True, limite le chargement aux 200 premières images.

        Returns
        -------
        list of np.ndarray
            Liste de tranches 2D en float32 (intensités brutes).

        Raises
        ------
        FileNotFoundError
            Si aucun fichier .tif n'est trouvé dans le dossier.
        """
        files = sorted(
            [
                f for f in self.folder.iterdir()
                if f.suffix.lower() in {".tif", ".tiff"} and not f.name.startswith(".")
            ],
            key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)],
        )

        if not files:
            raise FileNotFoundError(f"Aucun fichier TIF trouvé dans : {self.folder}")

        if fast_test and len(files) > 500:
            files = files[:500]
            logger.info("FAST TEST ACTIVÉ : chargement limité aux 500 premières images.")

        logger.info(f"Lecture de {len(files)} images TIF (intensités brutes)…")
        slices: list[np.ndarray] = []
        total = len(files)

        for i, f in enumerate(files, 1):
            slices.append(self._read_raw(f))
            if i % 100 == 0 or i == total:
                print(f"  Chargement : {i}/{total} images...")
                logger.debug(f"{i}/{total} images lues.")

        return slices

    def _read_raw(self, path: Path) -> np.ndarray:
        """
        Lit un fichier .tif et retourne les intensités en float32
        SANS normalisation. Gère les cas multi-page et RGB.

        Parameters
        ----------
        path : Path
            Chemin vers un fichier .tif.

        Returns
        -------
        np.ndarray
            Tableau 2D float32 avec les intensités brutes.
        """
        try:
            img = tifffile.imread(str(path))
        except Exception as e:
            logger.warning(f"Impossible de lire {path.name} : {e}. Tranche ignorée (zéros).")
            return None

        # Cas multi-page : garder la première page
        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            img = img[0]

        # Cas RGB : convertir en niveaux de gris (luminance)
        if img.ndim == 3:
            img = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])

        # Convertir en float32 pour uniformité, sans changer les valeurs
        return img.astype(np.float32)
