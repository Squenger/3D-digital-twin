import argparse
import os
from pathlib import Path
from main import TifVers3D

def batch_process():
    print("=== Traitement par lot d'échantillons TIF vers 3D ===")
    
    dossier_parent = input("Veuillez entrer le chemin du dossier contenant les dossiers d'échantillons : ").strip()
    dossier_cible = input("Veuillez entrer le chemin du dossier cible pour les sorties (.stl) : ").strip()

    parent_path = Path(dossier_parent)
    cible_path = Path(dossier_cible)

    if not parent_path.exists() or not parent_path.is_dir():
        print(f"Erreur : Le dossier source '{parent_path}' n'existe pas ou n'est pas un dossier.")
        return

    # Créer le dossier cible s'il n'existe pas
    cible_path.mkdir(parents=True, exist_ok=True)

    # Récupérer tous les sous-dossiers (échantillons)
    echantillons = [d for d in parent_path.iterdir() if d.is_dir()]

    if not echantillons:
        print(f"Aucun sous-dossier (échantillon) trouvé dans '{parent_path}'.")
        return

    print(f"\n{len(echantillons)} échantillons trouvés. Début du traitement par lot")

    # Paramètres par défaut de main.py
    seuil = None
    epaisseur_couche = 0.015
    taille_voxel = 0.015
    visualiser = False
    plein = False
    test_rapide = False

    for i, echantillon_dir in enumerate(echantillons, 1):
        nom_echantillon = echantillon_dir.name
        chemin_sortie = cible_path / f"{nom_echantillon}.stl"
        
        print(f"\n[{i}/{len(echantillons)}] Traitement de l'échantillon : {nom_echantillon}")
        print(f" -> Dossier d'entrée : {echantillon_dir}")
        print(f" -> Fichier de sortie : {chemin_sortie}")
        
        try:
            pipeline = TifVers3D(
                dossier_entree=echantillon_dir,
                chemin_sortie=chemin_sortie,
                seuil=seuil,
                epaisseur_couche=epaisseur_couche,
                taille_voxel=taille_voxel,
                visualiser=visualiser,
                plein=plein,
                test_rapide=test_rapide,
            )
            pipeline.executer()
            print(f" -> Modèle sauvegardé avec succès.")
        except Exception as e:
            print(f" -> Erreur lors du traitement de l'échantillon {nom_echantillon} : {e}")

    print("\n=== Traitement par lot terminé ! ===")

if __name__ == "__main__":
    batch_process()
