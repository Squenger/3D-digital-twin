"""
Convertit un fichier TXT de coordonnées X, Y, Z en une image 2.5D (niveaux de gris).

L'intensité lumineuse de chaque pixel est proportionnelle à la valeur Z.
Les coordonnées X, Y sont rasterisées sur une grille régulière.

Usage :
    python xyz_to_image.py chemin/vers/fichier.txt [-o image.png] [--resolution 512]
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_xyz(filepath: str) -> np.ndarray:
    """
    Charge un fichier TXT contenant des coordonnées X, Y, Z.

    Supporte les fichiers avec ou sans en-tête.
    Colonnes séparées par des espaces, tabulations ou virgules.

    Returns
    -------
    np.ndarray de shape (N, 3) — colonnes X, Y, Z.
    """
    # Tenter de charger en sautant la première ligne si c'est un header
    try:
        data = np.loadtxt(filepath)
    except ValueError:
        data = np.loadtxt(filepath, skiprows=1)

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Le fichier doit contenir au moins 3 colonnes (X, Y, Z). "
                         f"Shape lue : {data.shape}")
    return data[:, :3]


def xyz_to_image(data: np.ndarray, resolution_um: float = 50.0) -> np.ndarray:
    """
    Convertit un nuage de points X, Y, Z en image 2D (niveaux de gris),
    en respectant le ratio d'aspect physique de l'échantillon.

    Parameters
    ----------
    data : np.ndarray
        Tableau (N, 3) avec colonnes X, Y, Z (généralement en millimètres).
    resolution_um : float
        Résolution XY voulue en microns (ex: 50.0 µm par pixel).

    Returns
    -------
    image : np.ndarray
        Image 2D (float64) avec intensité proportionnelle à Z.
        Les pixels sans données valent 0.
    extent : tuple
        (x_min, x_max, y_min, y_max) pour l'affichage physique.
    """
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Conversion de la résolution de microns vers l'unité des données (ici on assume millimètres)
    # 50 µm = 0.050 mm
    res_mm = resolution_um / 1000.0

    # Tailles de la grille (nombre de pixels en X et Y)
    width_px = int(np.ceil(x_range / res_mm)) if x_range > 0 else 1
    height_px = int(np.ceil(y_range / res_mm)) if y_range > 0 else 1

    # Normaliser X, Y directement en indices pixels
    ix = ((x - x_min) / res_mm).astype(int)
    iy = ((y - y_min) / res_mm).astype(int)

    # Clipper par sécurité 
    ix = np.clip(ix, 0, width_px - 1)
    iy = np.clip(iy, 0, height_px - 1)

    # Créer l'image — moyenne des Z si plusieurs points par pixel
    image = np.zeros((height_px, width_px), dtype=np.float64)
    count = np.zeros((height_px, width_px), dtype=np.int32)

    # Accumuler (iy = ligne, ix = colonne)
    np.add.at(image, (iy, ix), z)
    np.add.at(count, (iy, ix), 1)

    # Moyenne là où il y a des points
    mask = count > 0
    image[mask] /= count[mask]

    # Normaliser l'intensité dans [0, 1]
    z_min, z_max = image[mask].min(), image[mask].max()
    if z_max > z_min:
        image[mask] = (image[mask] - z_min) / (z_max - z_min)

    extent = (x_min, x_max, y_min, y_max)
    return image, extent


def display_and_save(image: np.ndarray, extent: tuple,
                     output_path: str = None, title: str = "Image 2.5D", show_graphics: bool = True):
    """
    Affiche l'image et la sauvegarde optionnellement.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(image, cmap="gray", origin="lower", extent=extent)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Intensité (proportionnelle à Z)")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Image sauvegardée : {output_path}")

    if show_graphics:
        plt.show()
    else:
        plt.close(fig)


def main(input_path: str | Path, output_path: str | Path = None, resolution_um: float = 50.0, show_graphics: bool = True):
    input_path = Path(input_path)
    
    # Charger
    print(f"Chargement de {input_path} ...")
    data = load_xyz(input_path)
    print(f"  {len(data)} points chargés")
    print(f"  X : [{data[:,0].min():.3f}, {data[:,0].max():.3f}]")
    print(f"  Y : [{data[:,1].min():.3f}, {data[:,1].max():.3f}]")
    print(f"  Z : [{data[:,2].min():.3f}, {data[:,2].max():.3f}]")

    # Convertir
    image, extent = xyz_to_image(data, resolution_um=resolution_um)
    print(f"  Image : {image.shape[0]}x{image.shape[1]} pixels")

    # Nom de sortie par défaut
    if output_path is None:
        output_path = input_path.with_suffix(".png")
    else:
        output_path = Path(output_path)

    # Afficher et sauvegarder
    title = f"Image 2.5D — {input_path.name}"
    display_and_save(image, extent, str(output_path), title, show_graphics=show_graphics)


if __name__ == "__main__":
    
    # ── Configuration ──────────────────────────────────────────────────────────
    INPUT_PATH = "chemin/vers/fichier.txt"
    OUTPUT_PATH = None  # Si None, remplacé automatiquement par le même nom en .png
    RESOLUTION_UM = 50.0  # Résolution de la grille en microns (ex: 50 microns / pixel)
    SHOW_GRAPHICS = True
    
    main(INPUT_PATH, OUTPUT_PATH, RESOLUTION_UM, show_graphics=SHOW_GRAPHICS)
