"""
bounds_detector.py
-------------------
Détection des bornes 3D (Z support, Y/X tight crop) pour l'isolation
de la pièce dans un volume micro-CT.

Stratégie :
-----------
1. Sur un échantillon de tranches, appliquer multi-Otsu (comme le pipeline
   original slice_processor.py) pour obtenir des masques binaires.
2. Trouver la plus grande composante connexe de chaque masque (= pièce ou support).
3. Distinguer pièce vs support par l'aire relative de la CC :
   - Aire > SUPPORT_AREA_RATIO  → support (structure dense, remplit l'image)
   - Aire ≤ SUPPORT_AREA_RATIO  → pièce (objet isolé entouré de vide)
4. Retenir les tranches Z qui correspondent à la pièce (pas au support).
5. Sur la bbox union de ces tranches → bornes Y et X serrées.

Aucun fichier du projet original n'a été modifié.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

# ── Constante clé ─────────────────────────────────────────────────────────
# Si l'aire de la plus grande CC dépasse cette fraction de l'image totale,
# la tranche est classée « support » et exclue.
# (Calibré sur Sample 1 : support ~50 %, pièce ~6-8 %)
SUPPORT_AREA_RATIO = 0.20

# Marge en pixels ajoutée autour de la bbox détectée
CROP_MARGIN_PX = 5


class BoundsDetector:
    """
    Détecte les bornes physiques (Z, Y, X) d'un empilement de tranches
    micro-CT en utilisant le seuillage multi-Otsu + analyse de composantes.

    Parameters
    ----------
    support_area_ratio : float
        Fraction de l'image (0–1) au-delà de laquelle une tranche est
        classifiée comme support. Défaut : 0.20 (20 %).
    crop_margin : int
        Marge en pixels ajoutée autour de la bbox de la pièce. Défaut : 5.
    sample_every : int
        Échantillonne une tranche tous les N pour le profil Z.
        Les bornes Y/X sont calculées sur toutes les tranches « pièce ».
        Défaut : 10.
    """

    def __init__(
        self,
        support_area_ratio: float = SUPPORT_AREA_RATIO,
        crop_margin: int = CROP_MARGIN_PX,
        sample_every: int = 10,
    ) -> None:
        self.support_area_ratio = support_area_ratio
        self.crop_margin = crop_margin
        self.sample_every = sample_every

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def determine_bounds(
        self,
        slices: list[np.ndarray],
        progress_callback=None,
    ) -> tuple[int, int, int, int, int, int]:
        """
        Calcule les bornes (z1, z2, y1, y2, x1, x2) de la pièce.

        Parameters
        ----------
        slices : list of np.ndarray
            Empilement de tranches 2D float32 (intensités brutes).
        progress_callback : callable, optional
            Callback(pct, 100, message).

        Returns
        -------
        tuple[int, int, int, int, int, int]
            (z_start, z_end, y_start, y_end, x_start, x_end)
            dans le système de coordonnées de l'empilement d'entrée.
        """
        depth = len(slices)
        if depth == 0:
            return 0, 0, 0, 0, 0, 0

        h, w = slices[0].shape
        total_px = h * w

        # ── Étape 1 : normalisation globale ──────────────────────────────
        # On normalise par le max global (sur un échantillon) pour que
        # threshold_multiotsu travaille toujours sur une plage [0,255] cohérente.
        sample_step = max(1, depth // 30)
        sampled = [slices[i] for i in range(0, depth, sample_step) if slices[i] is not None]
        global_max = max(float(s.max()) for s in sampled) if sampled else 1.0
        if global_max == 0:
            global_max = 1.0

        logger.info(f"Max global (sur {len(sampled)} tranches) : {global_max:.0f}")

        # ── Étape 2 : profil Z ───────────────────────────────────────────
        # Évalue chaque tranche échantillonnée : pièce ou support ?
        if progress_callback:
            progress_callback(5, 100, "Analyse du profil Z (détection support)…")

        z_indices = list(range(0, depth, self.sample_every))
        if z_indices[-1] != depth - 1:
            z_indices.append(depth - 1)

        z_profile: list[dict] = []
        for rank, z in enumerate(z_indices):
            sl = slices[z]
            if sl is None:
                z_profile.append({"z": z, "is_piece": False, "bbox": None})
                continue

            binary, largest_ratio, bbox = self._classify_slice(sl, global_max, total_px)
            is_piece = (largest_ratio <= self.support_area_ratio) and (largest_ratio > 0)
            z_profile.append({"z": z, "is_piece": is_piece, "bbox": bbox, "ratio": largest_ratio})

            if progress_callback and rank % 5 == 0:
                pct = 5 + 30 * rank / len(z_indices)
                progress_callback(pct, 100, f"Profil Z : {z}/{depth}…")

        # ── Étape 3 : trouver z_start et z_end ───────────────────────────
        piece_z = [p["z"] for p in z_profile if p["is_piece"]]
        if not piece_z:
            logger.warning("Aucune tranche identifiée comme 'pièce'. Support ratio peut-être trop restrictif.")
            # Fallback : tout garder
            z_start, z_end = 0, depth - 1
        else:
            z_start = piece_z[0]
            z_end = piece_z[-1]

        logger.info(
            f"Z pièce détecté : [{z_start} – {z_end}] | "
            f"Support exclu : {depth - (z_end - z_start + 1)} tranches"
        )

        # ── Étape 4 : bbox Y/X serrée sur les tranches pièce ─────────────
        if progress_callback:
            progress_callback(40, 100, "Calcul de la bbox Y/X de la pièce…")

        # On utilise les bboxes déjà calculées dans le profil + quelques
        # tranches supplémentaires au milieu pour précision
        bboxes = [p["bbox"] for p in z_profile if p["is_piece"] and p.get("bbox")]

        # Complète avec quelques tranches intermédiaires (toutes les 5)
        extra_indices = list(range(z_start, z_end + 1, max(1, (z_end - z_start) // 20)))
        for z in extra_indices:
            if slices[z] is None:
                continue
            _, _, bbox = self._classify_slice(slices[z], global_max, total_px)
            if bbox is not None:
                bboxes.append(bbox)

        if bboxes:
            y_starts = [b[0] for b in bboxes]
            y_ends   = [b[2] for b in bboxes]
            x_starts = [b[1] for b in bboxes]
            x_ends   = [b[3] for b in bboxes]

            y_start = max(0,     min(y_starts) - self.crop_margin)
            y_end   = min(h - 1, max(y_ends)   + self.crop_margin)
            x_start = max(0,     min(x_starts) - self.crop_margin)
            x_end   = min(w - 1, max(x_ends)   + self.crop_margin)
        else:
            y_start, y_end = 0, h - 1
            x_start, x_end = 0, w - 1

        logger.info(
            f"Bornes finales :\n"
            f"  Z : [{z_start} – {z_end}]  ({z_end - z_start + 1} tranches)\n"
            f"  Y : [{y_start} – {y_end}]  ({y_end - y_start + 1} px → "
            f"{(y_end - y_start + 1) * 0.015:.2f} mm)\n"
            f"  X : [{x_start} – {x_end}]  ({x_end - x_start + 1} px → "
            f"{(x_end - x_start + 1) * 0.015:.2f} mm)"
        )

        if progress_callback:
            progress_callback(60, 100, "Bornes détectées.")

        return z_start, z_end, y_start, y_end, x_start, x_end

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _classify_slice(
        self,
        sl: np.ndarray,
        global_max: float,
        total_px: int,
    ) -> tuple[np.ndarray, float, Optional[tuple]]:
        """
        Binarise une tranche (multi-Otsu), trouve la plus grande CC
        et retourne (masque_binaire, aire_relative_CC, bbox_CC).

        La normalisation par `global_max` garantit des seuils Otsu cohérents
        entre toutes les tranches, quelle que soit leur intensité locale.

        Returns
        -------
        binary : np.ndarray uint8
        largest_ratio : float  (aire_CC / total_px)
        bbox : tuple (y_min, x_min, y_max, x_max) or None
        """
        # Normalisation → uint8 (même logique que tif_loader._read + _process)
        u8 = np.clip(sl / global_max * 255, 0, 255).astype(np.uint8)

        # Multi-Otsu à 3 classes (identique à slice_processor._process)
        try:
            thresholds = threshold_multiotsu(u8, classes=3)
            binary = (u8 > int(thresholds[1])).astype(np.uint8)
        except Exception:
            # Fallback si multi-Otsu échoue (tranche uniforme)
            binary = np.zeros_like(u8, dtype=np.uint8)
            return binary, 0.0, None

        if binary.sum() == 0:
            return binary, 0.0, None

        # Plus grande composante connexe
        labeled = label(binary, connectivity=2)
        props = regionprops(labeled)
        if not props:
            return binary, 0.0, None

        largest = max(props, key=lambda p: p.area)
        ratio = largest.area / total_px
        # bbox = (row_min, col_min, row_max, col_max)
        bbox = largest.bbox  # (y_min, x_min, y_max, x_max) en convention skimage

        return binary, ratio, bbox

    def get_z_profile(
        self,
        slices: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne le profil Z sous forme de deux tableaux NumPy.
        Utile pour visualisation / debugging.

        Returns
        -------
        z_indices : np.ndarray
        ratios : np.ndarray  (aire_CC / total_px pour chaque tranche échantillonnée)
        """
        depth = len(slices)
        h, w = slices[0].shape
        total_px = h * w

        sample_step = max(1, depth // 30)
        sampled = [slices[i] for i in range(0, depth, sample_step) if slices[i] is not None]
        global_max = max(float(s.max()) for s in sampled) if sampled else 1.0

        z_indices = list(range(0, depth, self.sample_every))
        ratios = []
        for z in z_indices:
            if slices[z] is None:
                ratios.append(0.0)
                continue
            _, r, _ = self._classify_slice(slices[z], global_max, total_px)
            ratios.append(r)

        return np.array(z_indices), np.array(ratios)

    def tight_crop_from_volume(
        self,
        volume: np.ndarray,
        progress_callback=None,
    ) -> tuple[int, int, int, int, int, int]:
        """
        Calcule des bornes serrées (z1, z2, y1, y2, x1, x2) directement
        depuis le volume 3D (Z, Y, X) float32 en utilisant des profils
        d'intensité moyenne — bien plus fiable que l'analyse tranche par tranche.

        Stratégie
        ---------
        1. Profil Z = intensité moyenne par tranche → exclut le vide.
        2. Détection du support = pic d'intensité aux extrémités de Z
           (le support est plus dense → intensité moyenne plus haute).
        3. Profils Y et X = projection de la moyenne sur les tranches pièce.
        4. Application d'une marge de `crop_margin` pixels.
        """
        nz, ny, nx = volume.shape

        # ── 1. Profil Z (intensité moyenne par tranche) ───────────────────
        if progress_callback:
            progress_callback(5, 100, "Profil Z (intensité moyenne par tranche)…")

        z_mean = np.array([float(volume[z].mean()) for z in range(nz)])

        # ── 2. Bornes Z de la matière (exclut les tranches quasi vides) ───
        matter_thresh = z_mean.max() * 0.05
        matter_idx = np.where(z_mean >= matter_thresh)[0]
        if len(matter_idx) == 0:
            z1_matter, z2_matter = 0, nz - 1
        else:
            z1_matter, z2_matter = int(matter_idx[0]), int(matter_idx[-1])

        # ── 3. Détection du support par spike d'intensité aux extrémités ──
        # Le support (dense, uniforme) a une intensité moyenne distinctement
        # plus haute que la pièce (AM : matière poreuse/variable).
        z_body = z_mean[z1_matter:z2_matter + 1]
        n_body = len(z_body)

        # Intensité typique de la pièce = médiane du tiers central
        lo, hi = n_body // 3, 2 * n_body // 3
        piece_typical = float(np.median(z_body[lo:hi])) if hi > lo else float(z_body.mean())
        # Seuil support : 1.8× l'intensité typique
        support_thresh = piece_typical * 1.8

        logger.info(
            f"Intensité typique pièce (Z) : {piece_typical:.0f}  "
            f"| Seuil support : {support_thresh:.0f}"
        )

        if progress_callback:
            progress_callback(25, 100, "Exclusion du support en Z…")

        # Avance z1 tant que la tranche ressemble à du support
        z1_adj = z1_matter
        for dz in range(n_body):
            if z_body[dz] >= support_thresh:
                z1_adj = z1_matter + dz + 1
            else:
                break

        # Recule z2 depuis la fin
        z2_adj = z2_matter
        for dz in range(n_body - 1, -1, -1):
            if z_body[dz] >= support_thresh:
                z2_adj = z1_matter + dz - 1
            else:
                break

        # Sécurité : au moins 1 tranche
        if z2_adj < z1_adj:
            z1_adj, z2_adj = z1_matter, z2_matter

        logger.info(
            f"Z pièce (crop serré) : [{z1_adj} – {z2_adj}]  "
            f"({z2_adj - z1_adj + 1} tranches) | "
            f"Support/vide exclus : {z1_adj - z1_matter} début, {z2_matter - z2_adj} fin"
        )

        # ── 4. Profils Y / X sur les tranches pièce uniquement ────────────
        if progress_callback:
            progress_callback(50, 100, "Projection Y/X sur la zone pièce…")

        piece_vol = volume[z1_adj:z2_adj + 1]   # vue — pas de copie

        y_mean = piece_vol.mean(axis=(0, 2))    # (Y,)
        x_mean = piece_vol.mean(axis=(0, 1))    # (X,)

        thresh_y = y_mean.max() * 0.05
        thresh_x = x_mean.max() * 0.05

        y_idx = np.where(y_mean >= thresh_y)[0]
        x_idx = np.where(x_mean >= thresh_x)[0]

        y1_raw = int(y_idx[0])  if len(y_idx) else 0
        y2_raw = int(y_idx[-1]) if len(y_idx) else ny - 1
        x1_raw = int(x_idx[0])  if len(x_idx) else 0
        x2_raw = int(x_idx[-1]) if len(x_idx) else nx - 1

        # ── 5. Marge ─────────────────────────────────────────────────────
        z1 = max(0,    z1_adj - self.crop_margin)
        z2 = min(nz-1, z2_adj + self.crop_margin)
        y1 = max(0,    y1_raw - self.crop_margin)
        y2 = min(ny-1, y2_raw + self.crop_margin)
        x1 = max(0,    x1_raw - self.crop_margin)
        x2 = min(nx-1, x2_raw + self.crop_margin)

        logger.info(
            f"Bornes serrées 3D :\n"
            f"  Z : [{z1} – {z2}]  ({z2 - z1 + 1} tranches → {(z2-z1+1)*0.015:.2f} mm)\n"
            f"  Y : [{y1} – {y2}]  ({y2 - y1 + 1} px → {(y2-y1+1)*0.015:.2f} mm)\n"
            f"  X : [{x1} – {x2}]  ({x2 - x1 + 1} px → {(x2-x1+1)*0.015:.2f} mm)"
        )

        if progress_callback:
            progress_callback(100, 100, "Bornes serrées calculées.")

        return z1, z2, y1, y2, x1, x2


if __name__ == "__main__":
    import os
    import sys

    # Configuration simple du logging pour voir les étapes
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Chemin par défaut pour le test (basé sur les fichiers vus précédemment)
    default_npz = "correlation/outputs/sample1_micro_ct.npz"
    npz_path = sys.argv[1] if len(sys.argv) > 1 else default_npz

    if not os.path.exists(npz_path):
        logger.error(f"Fichier non trouvé : {npz_path}")
        logger.info(f"Usage : python bounds_detector.py [chemin_vers_volume.npz]")
        sys.exit(1)

    logger.info(f"Chargement de {npz_path}...")
    try:
        data = np.load(npz_path)
        # On cherche la clé du volume dans le .npz
        keys = list(data.keys())
        vol_key = "volume" if "volume" in keys else keys[0]
        volume = data[vol_key]
        logger.info(f"Volume chargé : {volume.shape} {volume.dtype}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}")
        sys.exit(1)

    # Instanciation de l'outil
    detector = BoundsDetector()

    # Exécution de la détection robuste (tight_crop_from_volume)
    logger.info("Lancement de la détection (méthode robuste basée sur les profils d'intensité)...")
    bounds = detector.tight_crop_from_volume(volume)

    # Affichage du résultat final
    z1, z2, y1, y2, x1, x2 = bounds
    print("\n" + "="*50)
    print("  RÉSULTATS DE LA DÉTECTION DES BORNES")
    print("="*50)
    print(f"  Z (tranches) : {z1} à {z2+1}")
    print(f"  Y (pixels)   : {y1} à {y2+1}")
    print(f"  X (pixels)   : {x1} à {x2+1}")
    print("="*50)
    print(f"Indexing Python suggéré :")
    print(f"  volume_croppé = volume[{z1}:{z2+1}, {y1}:{y2+1}, {x1}:{x2+1}]")
    print("="*50 + "\n")
