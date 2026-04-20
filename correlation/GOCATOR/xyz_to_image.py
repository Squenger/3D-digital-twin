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


def xyz_to_image(data: np.ndarray, resolution: int = None) -> np.ndarray:
    """
    Convertit un nuage de points X, Y, Z en image 2D (niveaux de gris).

    Parameters
    ----------
    data : np.ndarray
        Tableau (N, 3) avec colonnes X, Y, Z.
    resolution : int or None
        Résolution de l'image de sortie (resolution x resolution).
        Si None, la résolution est déduite automatiquement depuis les données.

    Returns
    -------
    image : np.ndarray
        Image 2D (float64) avec intensité proportionnelle à Z.
        Les pixels sans données valent 0.
    extent : tuple
        (x_min, x_max, y_min, y_max) pour l'affichage.
    """
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Résolution automatique : estimer depuis la densité de points
    if resolution is None:
        n_points = len(x)
        resolution = int(np.sqrt(n_points))
        resolution = max(resolution, 64)  # minimum 64x64

    # Calculer les indices de pixel pour chaque point
    # Normaliser X, Y dans [0, resolution-1]
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    ix = ((x - x_min) / x_range * (resolution - 1)).astype(int)
    iy = ((y - y_min) / y_range * (resolution - 1)).astype(int)

    # Clipper par sécurité
    ix = np.clip(ix, 0, resolution - 1)
    iy = np.clip(iy, 0, resolution - 1)

    # Créer l'image — moyenne des Z si plusieurs points par pixel
    image = np.zeros((resolution, resolution), dtype=np.float64)
    count = np.zeros((resolution, resolution), dtype=np.int32)

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
                     output_path: str = None, title: str = "Image 2.5D"):
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

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Convertit un fichier XYZ en image 2.5D (niveaux de gris)")
    parser.add_argument("input", type=str, help="Fichier TXT d'entrée (X Y Z)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Chemin de l'image de sortie (PNG)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Résolution de l'image (par défaut : auto)")
    args = parser.parse_args()

    # Charger
    print(f"Chargement de {args.input} ...")
    data = load_xyz(args.input)
    print(f"  {len(data)} points chargés")
    print(f"  X : [{data[:,0].min():.3f}, {data[:,0].max():.3f}]")
    print(f"  Y : [{data[:,1].min():.3f}, {data[:,1].max():.3f}]")
    print(f"  Z : [{data[:,2].min():.3f}, {data[:,2].max():.3f}]")

    # Convertir
    image, extent = xyz_to_image(data, resolution=args.resolution)
    print(f"  Image : {image.shape[0]}x{image.shape[1]} pixels")

    # Nom de sortie par défaut
    if args.output is None:
        args.output = str(Path(args.input).with_suffix(".png"))

    # Afficher et sauvegarder
    title = f"Image 2.5D — {Path(args.input).name}"
    display_and_save(image, extent, args.output, title)


if __name__ == "__main__":
    main()
