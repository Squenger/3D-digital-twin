"""
rogner_images.py
----------------
Outil indépendant pour rogner (crop) un lot d'images.
1. Demande le chemin du dossier contenant les images.
2. Demande le numéro de l'image de référence pour dessiner la coupe.
3. Ouvre une fenêtre pour sélectionner la zone à rogner avec la souris.
4. Applique ce masque à toutes les images et les sauvegarde dans un dossier "nom_cropped".
"""

import sys
import re
from pathlib import Path
import numpy as np
import cv2
import tifffile


def main():
    print("=" * 60)
    print("            OUTIL DE ROGNAGE D'IMAGES EN LOT")
    print("=" * 60)

    # 1. Demander le dossier
    dossier_str = input("\n> Faites glisser ou entrez le chemin du dossier d'images : ").strip()
    if not dossier_str:
        return
        
    # Nettoyer les guillemets si glissé-déposé dans le terminal
    if dossier_str.startswith("'") and dossier_str.endswith("'"):
        dossier_str = dossier_str[1:-1]
    elif dossier_str.startswith('"') and dossier_str.endswith('"'):
        dossier_str = dossier_str[1:-1]
        
    dossier_str = dossier_str.replace("\\ ", " ")

    dossier = Path(dossier_str).resolve()
    if not dossier.exists() or not dossier.is_dir():
        print(f"\n✗ Erreur : Le dossier {dossier} est introuvable.")
        sys.exit(1)

    # 2. Lister les images
    extensions_valides = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    fichiers = sorted(
        [f for f in dossier.iterdir() if f.suffix.lower() in extensions_valides and not f.name.startswith('.')],
        key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", p.stem)]
    )

    if not fichiers:
        print(f"\n✗ Aucune image trouvée dans {dossier}")
        sys.exit(1)

    print(f"\n✓ {len(fichiers)} images trouvées.")

    # 3. Demander le numéro de l'image
    choix_num = input(f"> Quelle image utiliser pour calibrer le rognage ? (de 1 à {len(fichiers)}, par défaut 1) : ").strip()
    if not choix_num:
        idx_test = 0
    else:
        try:
            idx_test = int(choix_num) - 1
            if idx_test < 0 or idx_test >= len(fichiers):
                print(f"✗ Numéro invalide. Choix par défaut : 1")
                idx_test = 0
        except ValueError:
            print("✗ Entrée invalide. Choix par défaut : 1")
            idx_test = 0

    fichier_test = fichiers[idx_test]
    print(f"\nOuverture de l'image : {fichier_test.name}")

    # 4. Charger l'image de test
    if fichier_test.suffix.lower() in {".tif", ".tiff"}:
        img = tifffile.imread(str(fichier_test))
        # Normalisation pour affichage correct (sinon l'image risque d'être noire sur OpenCV)
        if img.dtype != np.uint8:
            img_visu = (img / img.max() * 255).astype(np.uint8) if img.max() > 0 else img.astype(np.uint8)
        else:
            img_visu = img.copy()
    else:
        img_visu = cv2.imread(str(fichier_test), cv2.IMREAD_UNCHANGED)

    # Convertir en RGB pour l'affichage couleur du cadre OpenCV si l'image est 2D
    if len(img_visu.shape) == 2:
        img_visu = cv2.cvtColor(img_visu, cv2.COLOR_GRAY2BGR)

    # 5. Sélection de la zone
    print("\n" + "-" * 60)
    print("INSTRUCTIONS :")
    print("1. Dessinez un rectangle avec la souris.")
    print("2. Appuyez sur ENTREE ou ESPACE pour valider et commencer le rognage.")
    print("3. Appuyez sur 'c' pour annuler la sélection.")
    print("-" * 60)

    titre_fenetre = f"Rogner : {fichier_test.name} (Entree pour valider)"
    cv2.namedWindow(titre_fenetre, cv2.WINDOW_NORMAL)
    
    # Redimensionnement de la fenêtre pour qu'elle tienne sur l'écran si l'image est énorme
    hauteur_ecran = 800  # dimension arbitraire confortable
    if img_visu.shape[0] > hauteur_ecran:
        ratio = hauteur_ecran / float(img_visu.shape[0])
        largeur = int(img_visu.shape[1] * ratio)
        cv2.resizeWindow(titre_fenetre, largeur, hauteur_ecran)
        
    roi = cv2.selectROI(titre_fenetre, img_visu, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Force la fermeture de la fenêtre sur macOS

    x, y, w, h = [int(v) for v in roi]
    if w == 0 or h == 0:
        print("\n✗ Rognage annulé (zone sélectionnée vide).")
        sys.exit(0)

    print(f"\n✓ Zone de découpe validée : X={x}, Y={y}, Largeur={w}, Hauteur={h}")

    # 6. Création du dossier '_cropped'
    dossier_sortie = dossier.parent / f"{dossier.name}_cropped"
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    print(f"\nCréation du dossier de sauvegarde :\n> {dossier_sortie}")

    # 7. Application à toutes les images
    print("\nLancement du rognage...")
    
    for i, f in enumerate(fichiers, 1):
        # Lecture
        if f.suffix.lower() in {".tif", ".tiff"}:
            image = tifffile.imread(str(f))
        else:
            image = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            
        # Rognage (Attention: sur Numpy, l'ordre est Y, X)
        image_cropped = image[y:y+h, x:x+w]
        
        # Sauvegarde
        chemin_sortie = dossier_sortie / f.name
        if f.suffix.lower() in {".tif", ".tiff"}:
            tifffile.imwrite(str(chemin_sortie), image_cropped)
        else:
            cv2.imwrite(str(chemin_sortie), image_cropped)
            
        if i % 100 == 0 or i == len(fichiers):
            print(f"  {i}/{len(fichiers)} images rognées et sauvegardées...")
            
    print(f"\n✓ OPÉRATION TERMINÉE ! Les images sont dans : {dossier_sortie.name}")


if __name__ == "__main__":
    main()
