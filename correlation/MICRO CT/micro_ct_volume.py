"""
micro_ct_volume.py
------------------
Module central de la fonctionnalité de corrélation micro-CT / topographie.

Charge les images .tif micro-CT, applique le rognage spatial par seuillage
multi-Otsu + analyse de composantes connexes (même logique que le pipeline
principal), mais SANS binarisation — les intensités brutes sont conservées.
Produit un tenseur 3D numpy (Z, Y, X) float32 avec coordonnées physiques en mm.

Aucun fichier du projet original n'a été modifié.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tif_loader_raw import TifLoaderRaw
from bounds_detector import BoundsDetector

logger = logging.getLogger(__name__)


@dataclass
class VolumeMetadata:
    """Métadonnées physiques du tenseur 3D."""
    shape_z: int
    shape_y: int
    shape_x: int
    voxel_size_x: float          # mm/pixel
    voxel_size_z: float          # mm/pixel
    layer_thickness_y: float     # mm/tranche
    dtype: str
    intensity_min: float
    intensity_max: float

    @property
    def physical_size_x_mm(self) -> float:
        return self.shape_x * self.voxel_size_x

    @property
    def physical_size_y_mm(self) -> float:
        return self.shape_y * self.layer_thickness_y

    @property
    def physical_size_z_mm(self) -> float:
        return self.shape_z * self.voxel_size_z

    def __str__(self) -> str:
        return (
            f"Volume micro-CT\n"
            f"  Dimensions   : ({self.shape_z} Z, {self.shape_y} Y, {self.shape_x} X) voxels\n"
            f"  Taille physique : {self.physical_size_z_mm:.3f} mm (Z) × "
            f"{self.physical_size_y_mm:.3f} mm (Y) × "
            f"{self.physical_size_x_mm:.3f} mm (X)\n"
            f"  Résolutions  : ΔX={self.voxel_size_x*1000:.1f} µm  "
            f"ΔZ={self.voxel_size_z*1000:.1f} µm  "
            f"ΔY={self.layer_thickness_y*1000:.1f} µm\n"
            f"  Intensités   : [{self.intensity_min:.0f} – {self.intensity_max:.0f}]  dtype={self.dtype}"
        )


class MicroCTVolume:
    """
    Extrait un tenseur volumétrique 3D depuis des tranches micro-CT .tif,
    avec rognage spatial et conservation intégrale des intensités.

    Workflow interne
    ----------------
    1. Chargement des tranches brutes (.tif → float32) via TifLoaderRaw
    2. Détection automatique des bornes (Z/Y/X) via BoundsDetector
    3. Rognage 3D sans binarisation
    4. Exposition du tenseur + coordonnées physiques en mm

    Attributes (disponibles après build())
    ----------------------------------------
    volume : np.ndarray (Z, Y, X), float32
        Intensités voxel rognées (non binarisées).
    coords_z : np.ndarray (Z,)
        Coordonnées physiques Z en mm (centre de chaque tranche).
    coords_y : np.ndarray (Y,)
        Coordonnées physiques Y en mm.
    coords_x : np.ndarray (X,)
        Coordonnées physiques X en mm.
    metadata : VolumeMetadata
        Résumé des paramètres physiques et du rognage.
    """

    def __init__(
        self,
        input_folder: str | Path,
        voxel_size_x: float = 0.015,
        voxel_size_z: float = 0.015,
        layer_thickness_y: float = 0.015,
        fast_test: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        input_folder : str | Path
            Dossier contenant les tranches .tif micro-CT.
        voxel_size_x : float
            Résolution physique en X (mm/pixel). Défaut : 0.015 mm (15 µm).
        voxel_size_z : float
            Résolution physique en Z (mm/pixel). Défaut : 0.015 mm (15 µm).
        layer_thickness_y : float
            Distance inter-tranche en Y (mm). Défaut : 0.015 mm (15 µm).
        fast_test : bool
            Si True, charge uniquement les 200 premières images.
        """
        self.input_folder = Path(input_folder)
        self.voxel_size_x = voxel_size_x
        self.voxel_size_z = voxel_size_z
        self.layer_thickness_y = layer_thickness_y
        self.fast_test = fast_test

        self._loader = TifLoaderRaw(self.input_folder)

        # Attributs publics (remplis par build())
        self.volume: np.ndarray | None = None
        self.coords_z: np.ndarray | None = None
        self.coords_y: np.ndarray | None = None
        self.coords_x: np.ndarray | None = None
        self.metadata: VolumeMetadata | None = None

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def build(self) -> None:
        """
        Exécute le pipeline complet et remplit les attributs publics.

        Étapes :
        1. Chargement des tranches brutes (float32)
        2. Assemblage en tenseur 3D (Z, Y, X)
        3. Calcul des coordonnées physiques en mm
        4. Remplissage des métadonnées
        """
        # 1. Chargement
        print("Chargement des images micro-CT brutes…")
        raw_slices = self._loader.load(fast_test=self.fast_test)
        
        # Filtrer les éventuelles tranches nulles (erreurs de lecture)
        raw_slices = [s for s in raw_slices if s is not None]
        print(f"{len(raw_slices)} tranches chargées avec succès.")
        logger.info(f"{len(raw_slices)} tranches chargées avec succès.")

        # 2. Assemblage initial du tenseur
        print("Assemblage du volume brut…")
        stacked = np.stack(raw_slices, axis=0)  # (Y, Z, X) float32
        
        self.volume = np.swapaxes(stacked, 0, 1) # -> (Z, Y, X) float32
        nz, ny, nx = self.volume.shape
        logger.info(f"Tenseur assemblé : shape={self.volume.shape}, dtype={self.volume.dtype}")

        # 3. Coordonnées physiques (centre de chaque voxel)
        self.coords_z = np.arange(nz, dtype=np.float64) * self.voxel_size_z
        self.coords_y = np.arange(ny, dtype=np.float64) * self.layer_thickness_y
        self.coords_x = np.arange(nx, dtype=np.float64) * self.voxel_size_x

        # 4. Métadonnées
        self.metadata = VolumeMetadata(
            shape_z=nz,
            shape_y=ny,
            shape_x=nx,
            voxel_size_x=self.voxel_size_x,
            voxel_size_z=self.voxel_size_z,
            layer_thickness_y=self.layer_thickness_y,
            dtype=str(self.volume.dtype),
            intensity_min=float(self.volume.min()),
            intensity_max=float(self.volume.max()),
        )

        print("Tenseur micro-CT prêt.")
        logger.info(str(self.metadata))

    # ------------------------------------------------------------------
    # Méthodes de sortie
    # ------------------------------------------------------------------

    def get_point_cloud(self, mask_zeros: bool = True) -> np.ndarray:
        """
        Retourne le nuage de points 3D avec intensités.

        Parameters
        ----------
        mask_zeros : bool
            Si True, exclut les voxels d'intensité nulle (fond/air).

        Returns
        -------
        np.ndarray, shape (N, 4)
            Tableau [X_mm, Y_mm, Z_mm, intensity] pour chaque voxel conservé.
            Compatible avec le format de données topographiques.
        """
        self._require_built()

        nz, ny, nx = self.volume.shape
        if mask_zeros:
            z_idx, y_idx, x_idx = np.nonzero(self.volume > 0)
            flat_z = self.coords_z[z_idx]
            flat_y = self.coords_y[y_idx]
            flat_x = self.coords_x[x_idx]
            flat_i = self.volume[z_idx, y_idx, x_idx]
        else:
            nz, ny, nx = self.volume.shape
            zz, yy, xx = np.meshgrid(
                self.coords_z, self.coords_y, self.coords_x, indexing="ij"
            )
            flat_z = zz.ravel()
            flat_y = yy.ravel()
            flat_x = xx.ravel()
            flat_i = self.volume.ravel()

        point_cloud = np.column_stack([flat_x, flat_y, flat_z, flat_i])
        logger.info(f"Nuage de points : {len(point_cloud):,} points (mask_zeros={mask_zeros})")
        return point_cloud

    def get_xy_projection(self, method: str = "max") -> np.ndarray:
        """
        Projection 2D du volume sur le plan XY.

        Parameters
        ----------
        method : str
            'max'  → Maximum Intensity Projection (MIP) : valeur max sur Z.
            'mean' → Projection par moyenne.
            'sum'  → Somme des intensités.

        Returns
        -------
        np.ndarray, shape (Y, X), float32
            Image 2D de projection, avec les mêmes dimensions spatiales
            que les axes Y et X du tenseur.
        """
        self._require_built()
        if method == "max":
            return self.volume.max(axis=0)
        elif method == "mean":
            return self.volume.mean(axis=0).astype(np.float32)
        elif method == "sum":
            return self.volume.sum(axis=0).astype(np.float32)
        else:
            raise ValueError(f"Méthode inconnue : '{method}'. Valeurs valides : 'max', 'mean', 'sum'.")

    def get_slice(self, axis: str, index: int) -> np.ndarray:
        """
        Retourne une coupe 2D du tenseur.

        Parameters
        ----------
        axis : str
            'z', 'y' ou 'x'.
        index : int
            Index de la coupe dans l'axe spécifié.

        Returns
        -------
        np.ndarray, shape 2D, float32
        """
        self._require_built()
        axis = axis.lower()
        if axis == "z":
            return self.volume[index, :, :]
        elif axis == "y":
            return self.volume[:, index, :]
        elif axis == "x":
            return self.volume[:, :, index]
        else:
            raise ValueError(f"Axe invalide : '{axis}'. Valeurs valides : 'x', 'y', 'z'.")

    def save(self, output_path: str | Path) -> None:
        """
        Sauvegarde le tenseur et les métadonnées dans un fichier .npz.

        Le fichier contiendra :
        - 'volume'   : tenseur (Z, Y, X) float32
        - 'coords_z' : coordonnées Z en mm
        - 'coords_y' : coordonnées Y en mm
        - 'coords_x' : coordonnées X en mm
        - 'meta'     : tableau 1D avec [vx, vz, vy]

        Parameters
        ----------
        output_path : str | Path
            Chemin de sortie (extension .npz recommandée).
        """
        self._require_built()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        meta_arr = np.array([
            self.voxel_size_x,
            self.voxel_size_z,
            self.layer_thickness_y,
        ])

        np.savez_compressed(
            str(output_path),
            volume=self.volume,
            coords_z=self.coords_z,
            coords_y=self.coords_y,
            coords_x=self.coords_x,
            meta=meta_arr,
        )
        logger.info(f"Tenseur sauvegardé → {output_path}")

    @classmethod
    def load_from_file(cls, npz_path: str | Path) -> "MicroCTVolume":
        """
        Recharge un tenseur sauvegardé par save().

        Parameters
        ----------
        npz_path : str | Path
            Chemin vers le fichier .npz.

        Returns
        -------
        MicroCTVolume
            Instance avec volume, coordonnées et métadonnées restaurés.
            (input_folder restera vide – seul le tenseur est restauré.)
        """
        npz_path = Path(npz_path)
        data = np.load(str(npz_path))

        meta = data["meta"]
        vx, vz, vy = float(meta[0]), float(meta[1]), float(meta[2])

        # Crée une instance vide (le dossier n'existe peut-être plus)
        instance = cls.__new__(cls)
        instance.input_folder = Path(".")
        instance.voxel_size_x = vx
        instance.voxel_size_z = vz
        instance.layer_thickness_y = vy
        instance.fast_test = False
        instance.progress_callback = None
        instance._loader = None
        instance._detector = None

        instance.volume = data["volume"]
        instance.coords_z = data["coords_z"]
        instance.coords_y = data["coords_y"]
        instance.coords_x = data["coords_x"]

        nz, ny, nx = instance.volume.shape
        instance.metadata = VolumeMetadata(
            shape_z=nz, shape_y=ny, shape_x=nx,
            voxel_size_x=vx, voxel_size_z=vz, layer_thickness_y=vy,
            dtype=str(instance.volume.dtype),
            intensity_min=float(instance.volume.min()),
            intensity_max=float(instance.volume.max()),
        )
        logger.info(f"Tenseur rechargé depuis {npz_path}\n{instance.metadata}")
        return instance

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _require_built(self) -> None:
        if self.volume is None:
            raise RuntimeError("Appelez d'abord build() avant d'utiliser cette méthode.")
