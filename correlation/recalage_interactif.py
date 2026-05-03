import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import open3d as o3d
from skimage import filters
import tifffile

# On importe les fonctions d'ali2.py
from ali2 import (
    CONFIG,
    charger_couches_ply,
    charger_ct,
    aligner,
    recadrer_ct,
    extraire_tranches
)

matplotlib.use('TkAgg') # Use interactive backend

class RecalageInteractif:
    def __init__(self, resultats, vol_ct_full, res_xy, res_z, dx_init, dy_init, dz_init):
        self.resultats = resultats
        self.vol_ct_full = vol_ct_full
        self.res_xy = res_xy
        self.res_z = res_z
        
        self.dx_init = dx_init
        self.dy_init = dy_init
        self.dz_init = dz_init
        
        self.Nz_full, self.Ny_full, self.Nx_full = vol_ct_full.shape
        
        # Projections complètes XZ et YZ (calculées une seule fois)
        print("  Calcul des projections XZ et YZ complètes pour l'affichage...")
        # On peut sous-échantillonner légèrement le CT pour l'affichage si besoin,
        # mais les projections 2D sont rapides à afficher.
        self.proj_xz_full = vol_ct_full.max(axis=1)
        self.proj_yz_full = vol_ct_full.max(axis=2)
        
        nonzero_full = vol_ct_full[vol_ct_full > 0]
        if nonzero_full.size > 0:
            self.vmin_g, self.vmax_g = np.percentile(nonzero_full, [1, 99])
        else:
            self.vmin_g, self.vmax_g = 0, 255
            
        self.ext_z = [0, self.Nz_full * self.res_z]
        
        # Offsets fins pour chaque couche: dict{couche_id: {'dx': 0.0, 'dy': 0.0, 'dz': 0.0}}
        self.offsets_fins = {res['couche_id']: {'dx': 0.0, 'dy': 0.0, 'dz': 0.0} for res in resultats}
        
        self.current_layer_idx = 0
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 7))
        plt.subplots_adjust(bottom=0.3) # Espace pour les sliders et boutons
        
        # --- UI Elements ---
        axcolor = 'lightgoldenrodyellow'
        self.ax_dx = plt.axes([0.2, 0.15, 0.5, 0.03], facecolor=axcolor)
        self.ax_dy = plt.axes([0.2, 0.10, 0.5, 0.03], facecolor=axcolor)
        self.ax_dz = plt.axes([0.2, 0.05, 0.5, 0.03], facecolor=axcolor)
        
        self.slider_dx = Slider(self.ax_dx, 'Delta X (mm)', -2.0, 2.0, valinit=0.0, valstep=self.res_xy)
        self.slider_dy = Slider(self.ax_dy, 'Delta Y (mm)', -2.0, 2.0, valinit=0.0, valstep=self.res_xy)
        self.slider_dz = Slider(self.ax_dz, 'Delta Z (mm)', -1.0, 1.0, valinit=0.0, valstep=self.res_z)
        
        self.slider_dx.on_changed(self.update_plot)
        self.slider_dy.on_changed(self.update_plot)
        self.slider_dz.on_changed(self.update_plot)
        
        self.ax_prev = plt.axes([0.1, 0.22, 0.1, 0.04])
        self.ax_next = plt.axes([0.8, 0.22, 0.1, 0.04])
        self.ax_save = plt.axes([0.8, 0.05, 0.1, 0.04])
        
        self.btn_prev = Button(self.ax_prev, 'Précédent')
        self.btn_next = Button(self.ax_next, 'Suivant')
        self.btn_save = Button(self.ax_save, 'Tout Sauvegarder', color='lightgreen')
        
        self.btn_prev.on_clicked(self.prev_layer)
        self.btn_next.on_clicked(self.next_layer)
        self.btn_save.on_clicked(self.save_all)
        
        # Variables pour stocker les objets graphiques
        self.img_xy = None
        self.sc_xy = None
        self.img_xz = None
        self.sc_xz = None
        self.img_yz = None
        self.sc_yz = None
        
        self.bandeau_xz = None
        self.bandeau_yz = None
        
        # Afficher la première couche
        self.draw_layer()
        plt.show()

    def get_current_res(self):
        return self.resultats[self.current_layer_idx]

    def draw_layer(self):
        res = self.get_current_res()
        couche_id = res["couche_id"]
        
        # Récupérer les offsets fins courants pour cette couche
        off = self.offsets_fins[couche_id]
        
        # Mettre à jour silencieusement les sliders (sans déclencher update_plot)
        self.slider_dx.eventson = False
        self.slider_dx.set_val(off['dx'])
        self.slider_dx.eventson = True
        
        self.slider_dy.eventson = False
        self.slider_dy.set_val(off['dy'])
        self.slider_dy.eventson = True
        
        self.slider_dz.eventson = False
        self.slider_dz.set_val(off['dz'])
        self.slider_dz.eventson = True
        
        # Tout effacer
        for ax in self.axes:
            ax.clear()
            
        pts_g = res["pts_gocator"]
        chunk = res["chunk_ct"]
        vox_b = res["vox_bas"]
        vox_h = res["vox_haut"]
        
        dx_tot = self.dx_init + off['dx']
        dy_tot = self.dy_init + off['dy']
        dz_tot = self.dz_init + off['dz']

        self.fig.suptitle(f"Couche {couche_id} ({self.current_layer_idx+1}/{len(self.resultats)}) - Ajustez les sliders", fontsize=14)

        proj_xy = chunk.max(axis=0)
        nonzero = chunk[chunk > 0]
        vmin_l = np.percentile(nonzero, 1) if nonzero.size > 0 else 0
        vmax_l = np.percentile(nonzero, 99) if nonzero.size > 0 else 255

        ct_x0 = dx_tot
        ct_x1 = dx_tot + self.Nx_full * self.res_xy
        ct_y0 = dy_tot
        ct_y1 = dy_tot + self.Ny_full * self.res_xy

        # --- XY ---
        self.img_xy = self.axes[0].imshow(proj_xy, cmap='gray', origin='lower',
                                          extent=[ct_x0, ct_x1, ct_y0, ct_y1],
                                          vmin=vmin_l, vmax=vmax_l, aspect='equal')
        self.axes[0].set_title(f'Couche {couche_id} — XY')
        if pts_g.shape[0] > 0:
            z_vals = pts_g[:, 2]
            self.sc_xy = self.axes[0].scatter(pts_g[:, 0], pts_g[:, 1],
                                              c=z_vals, cmap='coolwarm', s=0.5, alpha=0.9)
            
        # --- XZ ---
        self.img_xz = self.axes[1].imshow(self.proj_xz_full, cmap='gray', origin='lower',
                                          extent=[ct_x0, ct_x1, *self.ext_z],
                                          vmin=self.vmin_g, vmax=self.vmax_g, aspect='auto')
        z_bas_mm  = vox_b * self.res_z
        z_haut_mm = (vox_h + 1) * self.res_z
        self.bandeau_xz = self.axes[1].axhspan(z_bas_mm, z_haut_mm, color='yellow', alpha=0.2)
        self.axes[1].set_title(f'Couche {couche_id} — XZ')
        if pts_g.shape[0] > 0:
            self.sc_xz = self.axes[1].scatter(pts_g[:, 0], pts_g[:, 2] - dz_tot,
                                              c=z_vals, cmap='coolwarm', s=0.5, alpha=0.9)

        # --- YZ ---
        self.img_yz = self.axes[2].imshow(self.proj_yz_full, cmap='gray', origin='lower',
                                          extent=[ct_y0, ct_y1, *self.ext_z],
                                          vmin=self.vmin_g, vmax=self.vmax_g, aspect='auto')
        self.bandeau_yz = self.axes[2].axhspan(z_bas_mm, z_haut_mm, color='yellow', alpha=0.2)
        self.axes[2].set_title(f'Couche {couche_id} — YZ')
        if pts_g.shape[0] > 0:
            self.sc_yz = self.axes[2].scatter(pts_g[:, 1], pts_g[:, 2] - dz_tot,
                                              c=z_vals, cmap='coolwarm', s=0.5, alpha=0.9)
            
        self.fig.canvas.draw_idle()

    def update_plot(self, val):
        # Callback rapide quand un slider bouge
        res = self.get_current_res()
        couche_id = res["couche_id"]
        
        dx_fine = self.slider_dx.val
        dy_fine = self.slider_dy.val
        dz_fine = self.slider_dz.val
        
        # Mettre à jour l'état
        self.offsets_fins[couche_id]['dx'] = dx_fine
        self.offsets_fins[couche_id]['dy'] = dy_fine
        self.offsets_fins[couche_id]['dz'] = dz_fine
        
        dx_tot = self.dx_init + dx_fine
        dy_tot = self.dy_init + dy_fine
        dz_tot = self.dz_init + dz_fine
        
        ct_x0 = dx_tot
        ct_x1 = dx_tot + self.Nx_full * self.res_xy
        ct_y0 = dy_tot
        ct_y1 = dy_tot + self.Ny_full * self.res_xy
        
        if self.img_xy is not None:
            self.img_xy.set_extent([ct_x0, ct_x1, ct_y0, ct_y1])
        if self.img_xz is not None:
            self.img_xz.set_extent([ct_x0, ct_x1, *self.ext_z])
        if self.img_yz is not None:
            self.img_yz.set_extent([ct_y0, ct_y1, *self.ext_z])
            
        pts_g = res["pts_gocator"]
        if pts_g.shape[0] > 0:
            # XZ scatter update
            if self.sc_xz is not None:
                offsets_xz = np.c_[pts_g[:, 0], pts_g[:, 2] - dz_tot]
                self.sc_xz.set_offsets(offsets_xz)
            # YZ scatter update
            if self.sc_yz is not None:
                offsets_yz = np.c_[pts_g[:, 1], pts_g[:, 2] - dz_tot]
                self.sc_yz.set_offsets(offsets_yz)
                
        self.fig.canvas.draw_idle()

    def prev_layer(self, event):
        if self.current_layer_idx > 0:
            self.current_layer_idx -= 1
            self.draw_layer()

    def next_layer(self, event):
        if self.current_layer_idx < len(self.resultats) - 1:
            self.current_layer_idx += 1
            self.draw_layer()

    def save_all(self, event):
        print("\n Sauvegarde en cours ")
        os.makedirs("verif_couches_interactif", exist_ok=True)
        
        # 1. Sauvegarde des images actuelles pour vérification
        print("  Génération des PNG")
        
        # Mémoriser l'état actuel
        idx_mem = self.current_layer_idx
        
        for i in range(len(self.resultats)):
            self.current_layer_idx = i
            self.draw_layer()
            plt.savefig(f"verif_couches_interactif/couche_{self.resultats[i]['couche_id']:02d}.png", dpi=150)
            
        # Restaurer l'état
        self.current_layer_idx = idx_mem
        self.draw_layer()
        
        # 2. Sauvegarde du fichier JSON des offsets (légère)
        json_data = {
            "dx_global_init": self.dx_init,
            "dy_global_init": self.dy_init,
            "dz_global_init": self.dz_init,
            "offsets_couches": self.offsets_fins
        }
        
        with open("offsets_recalage.json", "w") as f:
            json.dump(json_data, f, indent=4)
        print("  Sauvegardé : offsets_recalage.json")
        
        # 3. Sauvegarde du tenseur recadré (léger)
        np.savez_compressed("micro_ct_recale_interactif.npz", tensor=self.vol_ct_full)
        print("  Sauvegardé : micro_ct_recale_interactif.npz ")
        print(" Sauvegarde terminée ! Vous pouvez fermer la fenêtre ")


def main():
    cfg = CONFIG
    res_xy = cfg["RESOLUTION_CT_MM"]

    print("=" * 65)
    print("  RECALAGE MANUEL INTERACTIF GOCATOR <-> MICRO-CT")
    print("=" * 65)

    print("\n Chargement des couches PLY…")
    pts_ply, nb_couches = charger_couches_ply(cfg)

    print(f"\n Chargement CT depuis NPZ…")
    vol_ct, res_z = charger_ct(cfg)

    print("\nCalcul de l'alignement initial global…")
    dx, dy, dz = aligner(pts_ply, vol_ct, res_xy, res_z)

    print("\n Recadrage CT à la taille physique de l'objet…")
    vol_ct_crop, z_off, x_off, y_off = recadrer_ct(
        vol_ct, res_xy, res_z, cfg["TAILLE_OBJET_MM"]
    )
    
    dz_crop = dz + z_off * res_z
    dx_crop = dx + x_off * res_xy
    dy_crop = dy + y_off * res_xy

    print("\n Lancement de l'interface graphique…")
    resultats = extraire_tranches(
        vol_ct_crop, pts_ply, nb_couches,
        cfg["EPAISSEUR_COUCHE"], res_z, dz_crop
    )
    
    app = RecalageInteractif(resultats, vol_ct_crop, res_xy, res_z, dx_crop, dy_crop, dz_crop)

if __name__ == "__main__":
    main()
