import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from micro_ct_volume import MicroCTVolume
import islation_support

def show_slices(volume_raw, volume_processed, save_path: Path, show_graphics: bool):
    """
    Sauvegarde (et affiche au besoin) les coupes centrales ZY, ZX et YX du tenseur brut vs post-traité.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#121212")
    for ax in axes.flat:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="w")
        for s in ax.spines.values():
            s.set_edgecolor("#444")
            
    vols = [(volume_raw, "Volume Brut"), (volume_processed, "Volume Post-Traité")]
    
    for row, (vol, title) in enumerate(vols):
        if vol is None: continue
        z_mid, y_mid, x_mid = [s // 2 for s in vol.shape]
        
        # Coupe ZY (X central)
        im1 = axes[row, 0].imshow(vol[:, :, x_mid], cmap='gray', aspect='auto')
        axes[row, 0].set_title(f"{title} - Coupe ZY (X={x_mid})", color="w")
        
        # Coupe ZX (Y central)
        im2 = axes[row, 1].imshow(vol[:, y_mid, :], cmap='gray', aspect='auto')
        axes[row, 1].set_title(f"{title} - Coupe ZX (Y={y_mid})", color="w")
        
        # Coupe YX (Z central)
        im3 = axes[row, 2].imshow(vol[z_mid, :, :], cmap='gray', aspect='auto')
        axes[row, 2].set_title(f"{title} - Coupe YX (Z={z_mid})", color="w")

    plt.suptitle(f"Comparaison: Brut {volume_raw.shape} VS Post-traité {volume_processed.shape}", color="white", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_graphics:
        plt.show()
    plt.close(fig)

def main(sample_path: str | Path, output_dir: str | Path, show_graphics: bool = False):
    sample_path = Path(sample_path)
    output_dir = Path(output_dir)
    folder_name = sample_path.name
    
    print("=== Pipeline Micro-CT: Génération, Sauvegarde & Post-traitement ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger les images => tenseur 3D via MicroCTVolume (sans traitement)
    print(f"\n1. Chargement du volume brut depuis : {sample_path}")
    vol = MicroCTVolume(
        input_folder=sample_path,
        voxel_size_x=0.015,
        voxel_size_z=0.015,
        layer_thickness_y=0.015,
        fast_test=False
    )
    vol.build()
    
    raw_tensor = vol.volume
    if raw_tensor is None:
        print("Erreur: Le chargement a échoué.")
        return

    print(f"\n-> Tenseur brut généré : {raw_tensor.shape} (dtype: {raw_tensor.dtype})")
    
    # Sauvegarder tenseur brut
    path_raw = output_dir / f"tensor_{folder_name}_raw.npz"
    print(f"\n2. Sauvegarde du tenseur brut dans : {path_raw}")
    np.savez_compressed(path_raw, volume=raw_tensor)
    
    # Post-traitement via islation_support
    print("\n3. Post-traitement en cours...")
    
    print("   -> Suppression du support par variance")
    seuil_variance = 2200000  # à ajuster si besoin
    volume_propre, y_cut, _ = islation_support.remove_support_by_variance(raw_tensor, seuil_variance=seuil_variance)
    
    print("   -> Extraction Pièce (multi-Otsu)")
    mask = islation_support.extraction_pièce(volume_propre)
    
    print("   -> Masquage Convex Hull et Rognage spatial")
    processed_tensor = islation_support.convex_hull_mask(mask)
    
    print(f"\n-> Tenseur post-traité généré : {processed_tensor.shape}")

    # Sauvegarder tenseur post-traité
    path_processed = output_dir / f"tensor_{folder_name}_post_traitement.npz"
    print(f"\n4. Sauvegarde du tenseur post-traité dans : {path_processed}")
    np.savez_compressed(path_processed, volume=processed_tensor)
    
    # Sauvegarder 
    
    fig_path = output_dir / f"coupes_{folder_name}.png"
    print(f"\n5. Génération et sauvegarde de la figure comparative dans : {fig_path}")
    show_slices(raw_tensor, processed_tensor, save_path=fig_path, show_graphics=show_graphics)

if __name__ == "__main__":

    for i in range (9) :

        SAMPLE_PATH = "/Volumes/KINGSTON/DATA_MICRO_CT/sample " + str(i+1) + "/Sample " + str(i+1) + " 15um/S" + str(i+1) + "_15um_Original"
        OUTPUT_DIR = Path(__file__).parent / "outputs"
        
        main(SAMPLE_PATH, OUTPUT_DIR, show_graphics=False)
