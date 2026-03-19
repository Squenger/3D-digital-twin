"""
exportateur_modele.py
---------------------
Module gérant l'exportation et la visualisation interactive des modèles 3D générés.
"""

from pathlib import Path

import trimesh


class ExportateurModele:
    """
    exportation vers divers formats 3D standards (.stl, .obj, .ply, .glb)
    et d'affichage interactif pour validation visuelle.
    """

    FORMATS = {".stl", ".obj", ".ply", ".glb"}

    def exporter(self, maillage: trimesh.Trimesh, chemin: str | Path) -> Path:
        """
        Enregistre le maillage de l'objet sur le disque. Le format de fichier cible 
        est identifié automatiquement à partir de l'extension du chemin fourni.
        """
        chemin = Path(chemin).resolve()
        ext = chemin.suffix.lower()
        if ext not in self.FORMATS:
            raise ValueError(f"Format «{ext}» non supporté. Utilisez : {', '.join(sorted(self.FORMATS))}")

        print(f"\nDébut de la sauvegarde vers : {chemin.name}...")
        chemin.parent.mkdir(parents=True, exist_ok=True)

        # Traitement spécifique pour le format .glb (nécessite une encapsulation dans une Scene)
        # Pour les autres formats (STL, OBJ, etc.), l'exportation se fait directement sur le maillage
        if ext == ".glb":
            print("  > Format GLB détecté. Création de la scène...")
            donnees = trimesh.scene.Scene(geometry={"modele": maillage}).export(file_type="glb")
        else:
            print(f"  > Format {ext} détecté. Exportation en cours (long selon la taille)...")
            donnees = maillage.export(file_type=ext.lstrip("."))

        if isinstance(donnees, str):
            donnees = donnees.encode()
            
        print("  > Écriture sur le disque...")
        chemin.write_bytes(donnees)

        taille_mb = chemin.stat().st_size / (1024 * 1024)
        print(f"✓ Fichier {chemin.name} sauvegardé ({taille_mb:.1f} Mo).")
        return chemin

    @staticmethod
    def visualiser(maillage: trimesh.Trimesh) -> None:
        """
        Ouvre une interface 3D interactive.
        """
        print("\nOuverture de la fenêtre 3D interactive...")
        maillage.show()
