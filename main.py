"""
tif_vers_3d.py
--------------
Script principal
Il gère le chargement des images TIF, leur binarisation, la reconstruction 3D
et l'exportation du modèle final.

"""

import time
from pathlib import Path

from chargeur_tif import ChargeurTif
from constructeur_modele import ConstructeurModele
from exportateur_modele import ExportateurModele
from processeur_tranches import ProcesseurTranches


class TifVers3D:
    """
    Classe gérant la reconstruction 3D.
    
    Architecture des étapes :
        1. Chargement des données brutes (ChargeurTif)
        2. Prétraitement et binarisation   (ProcesseurTranches)
        3. Assemblage volumique (Maillage) (ConstructeurModele)
        4. Sauvegarde ou visualisation     (ExportateurModele)
    """

    def __init__(
        self,
        dossier_entree: str | Path,
        chemin_sortie: str | Path | None = None,
        seuil: int | None = None,
        epaisseur_couche: float = 0.1,
        taille_voxel: float = 0.1,
        visualiser: bool = False,
        plein: bool = False,
        test_rapide: bool = False,
    ) -> None:
        self.dossier_entree = Path(dossier_entree)
        self.chemin_sortie = Path(chemin_sortie) if chemin_sortie else None
        self.visualiser = visualiser
        self.plein = plein
        self.test_rapide = test_rapide

        # Initialisation des sous-modules
        self.chargeur    = ChargeurTif(self.dossier_entree)
        self.processeur  = ProcesseurTranches(seuil=seuil)
        self.constructeur = ConstructeurModele(epaisseur_couche, taille_voxel)
        self.exportateur = ExportateurModele()

    def executer(self) -> None:
        tranches = self.chargeur.charger(test_rapide=self.test_rapide)
        tranches_bin = self.processeur.traiter_lot(tranches)
        maillage = self.constructeur.construire(tranches_bin, plein=self.plein)
        if self.chemin_sortie:
            self.exportateur.exporter(maillage, self.chemin_sortie)
        if self.visualiser:
            self.exportateur.visualiser(maillage)



def main() -> None:

    
    # Entrées / Sorties 
    dossier_entree = '/Volumes/My Book/3D TWIN AIMINE/DATA MICRO CT/sample 2/Sample 2 15um/S2_15um_Original_cropped'
    chemin_sortie  = "/Volumes/My Book/3D TWIN AIMINE/DATA MICRO CT/sample 2/Sample 2 15um/OUTPUT/modele2.stl"  # Définir à None pour désactiver l'export sur disque
    
    # Rendu et Post-traitement 
    visualiser  = False  # Affiche la visionneuse 3D interactive après l'export
    plein       = False  # Génère un modèle volumique massif (True) ou surfacique (False)
    test_rapide = False   # (OPTION DE DEBUG) Teste uniquement sur les 500 premières tranches 
    
    # Paramètres algorithmiques et physiques 
    seuil            = None  # seuil de binarisation manuelle [0-255]. None = automatique (Otsu)
    epaisseur_couche = 0.015   # distance entre deux tranches successives en mm)
    taille_voxel     = 0.015   #résolution en x y (mm)

    try:
        pipeline = TifVers3D(
            dossier_entree=dossier_entree,
            chemin_sortie=chemin_sortie,
            seuil=seuil,
            epaisseur_couche=epaisseur_couche,
            taille_voxel=taille_voxel,
            visualiser=visualiser,
            plein=plein,
            test_rapide=test_rapide,
        )
        pipeline.executer()
        
    except Exception as e:
        print(f"\n✗ Erreur : {e}")

if __name__ == "__main__":
    main()
