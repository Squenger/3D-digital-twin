"""
ali2.py - Recalage Gocator PLY <-> Micro-CT
============================================

STRATÉGIE :
  - Couche PLY = surface supérieure de chaque couche imprimée.
  - Couche 0 = base de la pièce (Z=0 mm), couche N-1 = sommet (Z max).
  - Alignement physique pur : pas de rotation, uniquement translations.
      * Le plan supérieur du CT (Z_max_CT en mm) = Z de la dernière couche PLY.
      * Les centres XY des deux nuages sont superposés.
  - Les PLY sont la CIBLE (fixes). Le CT est déplacé vers eux.
  - L'axe vertical du tenseur CT (axe 0, Z croissant) doit être orienté
    dans le même sens que l'axe d'empilement des couches PLY.
  - Pour chaque couche PLY i, on extrait le bloc de voxels CT entre z(i-1) et z(i).

SORTIES :
  - verif_couches_3d.png : vue 3D par couche.
  - verif_alignment_2d.png : coupes 2D de superposition.
"""

import os
import glob
import re
import numpy as np
import tifffile
import open3d as o3d
from skimage import filters, measure
import matplotlib
matplotlib.use("Agg")  # Pas d'affichage interactif au début
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# CONFIGURATION — à adapter selon vos données
# =============================================================================
CONFIG = {
    # Tenseur CT pré-calculé (NPZ, clé 'tensor', shape ZYX, uint8)
    "FICHIER_CT_NPZ"    : "correlation/MicroCT2/reconstructed_tensor_sample5.npz",
    "RESOLUTION_CT_MM"  : 0.015,       # mm / voxel (isotrope, 15 µm)
    "DOSSIER_PLY"       : "sortie/GOCATOR/5",
    "RESOLUTION_PLY_XY" : 0.05,        # mm / unité dans les coords PLY
    "EPAISSEUR_COUCHE"  : 0.4,         # mm entre couches
    # Axe du tenseur (ZYX) qui correspond à l'axe vertical de la pièce.
    # 0 = premier axe (slices TIF), 1 ou 2 si besoin.
    "CT_AXEZ"           : 0,
    # True si voxel 0 = haut de la pièce (inverser Z)
    "CT_INVERSER_Z"     : False,
    # Rotation du CT autour de l'axe Z (nombre de quarts de tour CCW)
    # -1 = -90° (Clockwise), 1 = +90° (CCW), 2 = 180°, 0 = aucune.
    "CT_ROTATION_Z"     : 0,
    # Taille physique RÉELLE de l'objet [X_mm, Y_mm, Z_mm]
    "TAILLE_OBJET_MM"   : [10.0, 10.0, 2.0],
}

# =============================================================================
# 1. Chargement des couches PLY
# =============================================================================

def charger_couches_ply(cfg):
    pattern  = os.path.join(cfg["DOSSIER_PLY"], "couche_nuage_*.ply")
    fichiers = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
    )
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier PLY : {pattern}")

    print(f"  {len(fichiers)} couche(s) PLY.")
    tous = []
    for idx, f in enumerate(fichiers):
        pcd = o3d.io.read_point_cloud(f)
        pts = np.asarray(pcd.points)
        if pts.shape[0] == 0:
            continue

        # Calcul de la résolution actuelle en X (espacement médian entre points uniques)
        x_vals = np.unique(np.round(pts[:, 0], 5))
        if len(x_vals) > 1:
            res_actuelle = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
            facteur = cfg["RESOLUTION_PLY_XY"] / res_actuelle
        else:
            facteur = 1.0

        if idx == 0:
            print(f"    Résolution PLY détectée : {res_actuelle:.5f} mm/pt → "
                  f"facteur d'échelle XY = {facteur:.4f}")

        out = np.zeros((pts.shape[0], 4))
        out[:, 0] = pts[:, 0] * facteur
        out[:, 1] = pts[:, 1] * facteur
        # Le Gocator mesure la SURFACE SUPÉRIEURE de la couche idx.
        # On recentre la topographie réelle autour du haut physique de la couche.
        z_scaled = pts[:, 2] * facteur
        z_base   = (idx + 1) * cfg["EPAISSEUR_COUCHE"]   # TOP de la couche
        out[:, 2] = z_scaled - z_scaled.mean() + z_base  # topographie centrée sur z_top
        out[:, 3] = idx
        tous.append(out)
        print(f"    Couche {idx}: {pts.shape[0]:>7} pts | Z={out[0,2]:.2f} mm "
              f"| X=[{out[:,0].min():.2f},{out[:,0].max():.2f}] mm "
              f"| Y=[{out[:,1].min():.2f},{out[:,1].max():.2f}] mm")
    
    pts_all = np.vstack(tous)
    nb      = len(fichiers)
    print(f"  PLY : Z_min={pts_all[:,2].min():.2f} mm, Z_max={pts_all[:,2].max():.2f} mm")
    return pts_all, nb


# =============================================================================
# 2. Chargement CT (TIF) — axe Z ré-orienté selon CT_AXEZ
# =============================================================================

def charger_ct(cfg):
    """Charge le tenseur CT depuis le fichier NPZ pré-calculé."""
    chemin = cfg["FICHIER_CT_NPZ"]
    res    = cfg["RESOLUTION_CT_MM"]

    print(f"  Chargement NPZ : {chemin}")
    data = np.load(chemin)
    vol  = data[data.files[0]].astype(np.float32)  # shape attendue : (Nz, Ny, Nx)

    # Réorienter si l'axe Z physique n'est pas l'axe 0
    ax = cfg["CT_AXEZ"]
    if ax != 0:
        vol = np.moveaxis(vol, ax, 0)

    if cfg["CT_INVERSER_Z"]:
        vol = vol[::-1, :, :]

    # Appliquer la rotation Z demandée
    k_rot = cfg.get("CT_ROTATION_Z", 0)
    if k_rot != 0:
        print(f"  Rotation CT : {k_rot * 90}° (k={k_rot})")
        vol = np.rot90(vol, k=k_rot, axes=(1, 2))

    print(f"  Tenseur CT shape (Nz,Ny,Nx) = {vol.shape}")
    print(f"  Hauteur physique CT = {vol.shape[0]*res:.2f} mm "
          f"({vol.shape[0]} voxels × {res:.4f} mm)")
    return vol, res   # résolution Z = résolution CT (isotrope)


# =============================================================================
# 3. Alignement : translation pure + diagnostic
# =============================================================================

def aligner(pts_ply, vol_ct, res_xy, res_z):
    """
    Calcule la translation 3D (dx, dy, dz) pour superposer le CT sur le Gocator.

    CONTRAINTE PHYSIQUE :
      Le sommet de la MATIÈRE dans le CT (dernier voxel avec de la matière)
      doit coïncider avec le Z de la couche PLY la plus haute.

    Retourne : (dx, dy, dz) en mm.
    """
    Nz, Ny, Nx = vol_ct.shape

    # --- Trouver le Z du sommet de la matière dans le CT ---
    # Profil Z : nombre de voxels non nuls dans chaque tranche
    seuil_otsu = filters.threshold_otsu(vol_ct)
    z_profile   = (vol_ct > seuil_otsu).sum(axis=(1, 2))  # (Nz,)

    # Seuil : on considère qu'une tranche "contient de la matière"
    # si elle a plus de 1% du maximum de la courbe
    seuil_profil = z_profile.max() * 0.01
    tranches_avec_matiere = np.where(z_profile > seuil_profil)[0]

    if tranches_avec_matiere.size == 0:
        raise RuntimeError("Aucune matière détectée dans le CT (profil Z vide).")

    z_vox_top_ct = tranches_avec_matiere[-1]   # indice voxel du sommet
    z_vox_bot_ct = tranches_avec_matiere[0]    # indice voxel de la base
    z_ct_top_mm  = z_vox_top_ct * res_z
    z_ct_bot_mm  = z_vox_bot_ct * res_z
    z_ct_ctr_mm  = (z_ct_top_mm + z_ct_bot_mm) / 2.0

    # Centres XY du CT (on prend le centre physique de la pièce, pas du tenseur entier)
    cx_ct = (Nx - 1) * res_xy / 2.0
    cy_ct = (Ny - 1) * res_xy / 2.0

    # Bounding box PLY
    cx_ply    = (pts_ply[:, 0].max() + pts_ply[:, 0].min()) / 2.0
    cy_ply    = (pts_ply[:, 1].max() + pts_ply[:, 1].min()) / 2.0
    z_ply_max = pts_ply[:, 2].max()   # = (nb_couches - 1) * epaisseur

    dx = cx_ply - cx_ct
    dy = cy_ply - cy_ct
    dz = z_ply_max - z_ct_top_mm   # SOMMET MATIÈRE CT = SOMMET PLY

    print(f"  CT matière : voxels Z=[{z_vox_bot_ct}, {z_vox_top_ct}]"
          f" → [{z_ct_bot_mm:.3f}, {z_ct_top_mm:.3f}] mm "
          f"(hauteur pièce = {z_ct_top_mm - z_ct_bot_mm:.3f} mm)")
    print(f"  PLY : centre XY=({cx_ply:.2f}, {cy_ply:.2f}), Z_max={z_ply_max:.2f} mm")
    print(f"  Translation : dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f} mm")
    print(f"  Vérif : Z_top_CT_dans_PLY = {z_ct_top_mm + dz:.4f} mm (doit = {z_ply_max:.4f} mm)")

    return dx, dy, dz


# =============================================================================
# 3b. Recadrage CT selon la taille physique réelle de l'objet (10×10×2 mm)
# =============================================================================

def recadrer_ct(vol_ct, res_xy, res_z, taille_objet_mm):
    """
    Recadre le tenseur CT pour ne conserver que le volume correspondant à
    l'objet physique réel (ex: 10 × 10 × 2 mm).

    Contraintes :
      - Z : les N_z_objet voxels supérieurs de la matière (sommet = dernière couche PLY).
      - XY : fenêtre centrée sur le centre de masse XY de la matière.

    Retourne :
      vol_recadre  : sous-tenseur (Nz_obj, Ny_obj, Nx_obj)
      z_vox_debut  : décalage Z dans vol_ct (pour recalculer dz dans extraire_tranches)
      x_vox_debut  : décalage X
      y_vox_debut  : décalage Y
    """
    Nz, Ny, Nx = vol_ct.shape
    wx_mm, wy_mm, wz_mm = taille_objet_mm

    nx_obj = int(np.round(wx_mm / res_xy))
    ny_obj = int(np.round(wy_mm / res_xy))
    nz_obj = int(np.round(wz_mm / res_z))

    # Détecter le sommet et le centre de masse XY de la matière
    seuil      = filters.threshold_otsu(vol_ct)
    binaire    = (vol_ct > seuil)
    z_profile  = binaire.sum(axis=(1, 2))
    zz = np.where(z_profile > z_profile.max() * 0.01)[0]
    z_top = int(zz[-1])

    # Centre de masse XY sur les nz_obj tranches du sommet
    region_top = binaire[max(0, z_top - nz_obj):z_top + 1, :, :]
    xy_proj    = region_top.sum(axis=0)
    coords     = np.argwhere(xy_proj > 0)
    if coords.size > 0:
        cy_vox = int(np.round(coords[:, 0].mean()))
        cx_vox = int(np.round(coords[:, 1].mean()))
    else:
        cy_vox, cx_vox = Ny // 2, Nx // 2

    # Indices de recadrage Z (nz_obj voxels du sommet vers le bas)
    z_fin   = min(Nz,  z_top + 1)
    z_debut = max(0,   z_fin - nz_obj)

    # Indices de recadrage XY (centrés sur le centre de masse)
    x_debut = max(0,  cx_vox - nx_obj // 2)
    x_fin   = min(Nx, x_debut + nx_obj)
    if x_fin == Nx:
        x_debut = Nx - nx_obj

    y_debut = max(0,  cy_vox - ny_obj // 2)
    y_fin   = min(Ny, y_debut + ny_obj)
    if y_fin == Ny:
        y_debut = Ny - ny_obj

    vol_recadre = vol_ct[z_debut:z_fin, y_debut:y_fin, x_debut:x_fin]

    print(f"  Recadrage CT (contrainte {wx_mm}×{wy_mm}×{wz_mm} mm) :")
    print(f"    Z : voxels [{z_debut}, {z_fin}] → {vol_recadre.shape[0]} vox "
          f"= {vol_recadre.shape[0]*res_z:.3f} mm")
    print(f"    Y : voxels [{y_debut}, {y_fin}] → {vol_recadre.shape[1]} vox "
          f"= {vol_recadre.shape[1]*res_xy:.3f} mm")
    print(f"    X : voxels [{x_debut}, {x_fin}] → {vol_recadre.shape[2]} vox "
          f"= {vol_recadre.shape[2]*res_xy:.3f} mm")
    print(f"    Taille finale : {vol_recadre.shape[2]*res_xy:.2f} × "
          f"{vol_recadre.shape[1]*res_xy:.2f} × {vol_recadre.shape[0]*res_z:.2f} mm")

    return vol_recadre, z_debut, x_debut, y_debut


# =============================================================================
# 4. Extraction des tranches CT par couche
# =============================================================================

def extraire_tranches(vol_ct, pts_ply, nb_couches, epaisseur, res_z, dz):
    """
    Pour chaque couche PLY i :
      - Le plan supérieur de la couche i est à z_ply = i * epaisseur.
      - Dans l'espace CT, ce plan est à z_ct = z_ply - dz.
      - En voxels CT : voxel_z = z_ct / res_z.
      - On extrait le bloc de voxels entre couche i-1 et couche i.
    
    dz = décalage (espace PLY - espace CT natif) le long de Z.
    """
    Nz = vol_ct.shape[0]
    resultats = []

    def ply_z_to_vox(z_ply_mm):
        z_ct_mm = z_ply_mm - dz
        return int(np.round(z_ct_mm / res_z))

    for i in range(nb_couches):
        pts_couche = pts_ply[pts_ply[:, 3] == i, :3]
        # La surface PLY de la couche i = TOP de la couche i = (i+1) * epaisseur.
        # Le chunk CT : volume compris entre le TOP de la couche i-1 et le TOP de la couche i.
        #   z_haut = (i + 1) * epaisseur  → top de la couche i
        #   z_bas  = max(0, i * epaisseur) → top de la couche i-1 (= bas de la couche i)
        z_haut_mm = (i + 1) * epaisseur
        z_bas_mm  = max(0.0, i * epaisseur)

        vox_haut = np.clip(ply_z_to_vox(z_haut_mm), 0, Nz - 1)
        vox_bas  = np.clip(ply_z_to_vox(z_bas_mm),  0, Nz - 1)

        # Garantir l'ordre croissant
        if vox_bas > vox_haut:
            vox_bas, vox_haut = vox_haut, vox_bas

        chunk = vol_ct[vox_bas:vox_haut + 1, :, :]

        print(f"    Couche {i} : Z_ply=[{z_bas_mm:.2f},{z_haut_mm:.2f}] mm "
              f"-> voxels CT Z=[{vox_bas},{vox_haut}] | chunk={chunk.shape}")

        resultats.append({
            "couche_id"  : i,
            "pts_gocator": pts_couche,
            "chunk_ct"   : chunk,
            "vox_bas"    : vox_bas,
            "vox_haut"   : vox_haut,
        })

    return resultats


# =============================================================================
# 5. Visualisation 3D par couche
# =============================================================================

def visualiser_3d(resultats, res_xy, res_z, dx, dy, dz):
    n    = len(resultats)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(7 * cols, 6 * rows))
    fig.suptitle("Gocator (rouge) + CT sous-jacent (bleu) — par couche", fontsize=13)

    for res in resultats:
        i     = res["couche_id"]
        pts_g = res["pts_gocator"]
        chunk = res["chunk_ct"]
        vox_b = res["vox_bas"]

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.set_title(f"Couche {i}  (CT voxZ=[{vox_b},{res['vox_haut']}])", fontsize=10)

        # Gocator — rouge
        if pts_g.shape[0] > 0:
            pas = max(1, pts_g.shape[0] // 3000)
            ax.scatter(pts_g[::pas, 0], pts_g[::pas, 1], pts_g[::pas, 2],
                       c="red", s=1, alpha=0.9, label="Gocator")

        # CT — bleu (transformé dans l'espace PLY via dx, dy, dz)
        if chunk.size > 0 and chunk.max() > 0:
            seuil = filters.threshold_otsu(chunk)
            coords = np.argwhere(chunk > seuil)          # (N, 3) : Z, Y, X
            if coords.shape[0] > 0:
                coords[:, 0] += vox_b                     # Z global
                x_w = coords[:, 2] * res_xy + dx          # X world
                y_w = coords[:, 1] * res_xy + dy          # Y world
                z_w = coords[:, 0] * res_z  + dz          # Z world
                pas = max(1, coords.shape[0] // 5000)
                ax.scatter(x_w[::pas], y_w[::pas], z_w[::pas],
                           c="steelblue", s=0.5, alpha=0.2, label="CT")

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.view_init(elev=20, azim=45)
        if i == 0:
            ax.legend(markerscale=5)

    plt.tight_layout()
    plt.savefig("verif_couches_3d.png", dpi=120)
    print("  Sauvegardé : verif_couches_3d.png")


def visualiser_superposition_3d_totale(pts_ply, vol_ct, dx, dy, dz, res_xy, res_z):
    """
    Affiche une superposition 3D complète des deux modèles (subsamplés).
    pts_ply : tous les points Gocator (N, 4)
    vol_ct : tenseur CT recadré
    dx, dy, dz : offsets pour passer du CT (voxels) au monde PLY
    """
    print("  Génération de la superposition 3D totale...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Superposition Totale : Gocator (rouge) vs Micro-CT (bleu)")

    # 1. Gocator - Rouge (on vise ~20 000 points pour la forme)
    pas_g = max(1, pts_ply.shape[0] // 20000)
    ax.scatter(pts_ply[::pas_g, 0], pts_ply[::pas_g, 1], pts_ply[::pas_g, 2],
               c='red', s=0.8, alpha=0.5, label='Gocator')

    # 2. Micro-CT - Bleu (on vise ~50 000 points pour la forme)
    seuil = filters.threshold_otsu(vol_ct)
    coords = np.argwhere(vol_ct > seuil)
    if coords.size > 0:
        pas_ct = max(1, coords.shape[0] // 50000)
        x_w = coords[::pas_ct, 2] * res_xy + dx
        y_w = coords[::pas_ct, 1] * res_xy + dy
        z_w = coords[::pas_ct, 0] * res_z + dz
        ax.scatter(x_w, y_w, z_w, c='blue', s=0.4, alpha=0.15, label='Micro-CT')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(markerscale=5)
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig("verif_superposition_3d_totale.png", dpi=150)
    print("  Sauvegardé : verif_superposition_3d_totale.png")


def visualiser_couches_2d_separees(resultats, vol_ct_full, res_xy, res_z, dx, dy, dz):
    """
    Génère une image de vérification par couche (style cross_sections.png).
    - XY  : projection max du chunk courant → vue de dessus.
    - XZ/YZ : projection max du VOLUME COMPLET recadré → structure CT visible
              avec un bandeau horizontal indiquant la Z de la couche courante.
    Points Gocator colorés par altitude (coolwarm), sans sous-échantillonnage.
    """
    print("  Génération des figures 2D par couche (style imshow)...")
    os.makedirs("verif_couches", exist_ok=True)

    Nz_full, Ny_full, Nx_full = vol_ct_full.shape
    # Projections XZ et YZ du volume complet (calculées une seule fois)
    proj_xz_full = vol_ct_full.max(axis=1)   # (Nz_full, Nx_full)
    proj_yz_full = vol_ct_full.max(axis=2)   # (Nz_full, Ny_full)

    # Echelle de couleur globale pour cohérence entre couches
    nonzero_full = vol_ct_full[vol_ct_full > 0]
    if nonzero_full.size > 0:
        vmin_g, vmax_g = np.percentile(nonzero_full, [1, 99])
    else:
        vmin_g, vmax_g = 0, 255

    ext_x    = [0, Nx_full * res_xy]
    ext_y    = [0, Ny_full * res_xy]
    ext_z    = [0, Nz_full * res_z]

    for res in resultats:
        i     = res["couche_id"]
        pts_g = res["pts_gocator"]
        chunk = res["chunk_ct"]
        vox_b = res["vox_bas"]
        vox_h = res["vox_haut"]

        Nc, Ny_c, Nx_c = chunk.shape

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        fig.suptitle(
            f"Couche {i}  [CT gris + Gocator coolwarm]  "
            f"Z_CT=[{vox_b * res_z:.2f}, {(vox_h+1) * res_z:.2f}] mm",
            fontsize=13
        )

        # -------- XY : style identique à compare_rotation_z.png --------
        # CT et PLY sont tous deux dans le repère monde (coordonnées PLY)
        proj_xy = chunk.max(axis=0)   # (Ny_c, Nx_c)
        nonzero = chunk[chunk > 0]
        vmin_l = np.percentile(nonzero, 1) if nonzero.size > 0 else 0
        vmax_l = np.percentile(nonzero, 99) if nonzero.size > 0 else 255

        # Extent CT dans le repère monde PLY (comme compare_rotation_z)
        ct_x0 = dx;  ct_x1 = dx + Nx_full * res_xy
        ct_y0 = dy;  ct_y1 = dy + Ny_full * res_xy
        axes[0].imshow(proj_xy, cmap='gray', origin='lower',
                       extent=[ct_x0, ct_x1, ct_y0, ct_y1],
                       vmin=vmin_l, vmax=vmax_l, aspect='equal')
        axes[0].set_title(f'Couche {i} — XY dessus')
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('Y (mm)')
        if pts_g.shape[0] > 0:
            z_vals = pts_g[:, 2]
            z_p1, z_p99 = np.percentile(z_vals, [1, 99])
            # Points PLY en coordonnées monde, sans offset
            sc0 = axes[0].scatter(pts_g[:, 0], pts_g[:, 1],
                                  c=z_vals, cmap='coolwarm', s=0.3, alpha=0.9,
                                  vmin=z_p1, vmax=z_p99)
            cb0 = plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
            cb0.set_label(f'Z PLY [{z_p1:.3f}–{z_p99:.3f}] mm', fontsize=9)

        # -------- XZ : fond = volume complet, bandeau = chunk courant --------
        axes[1].imshow(proj_xz_full, cmap='gray', origin='lower',
                       extent=[ct_x0, ct_x1, *ext_z],
                       vmin=vmin_g, vmax=vmax_g, aspect='auto')
        z_bas_mm  = vox_b * res_z
        z_haut_mm = (vox_h + 1) * res_z
        axes[1].axhspan(z_bas_mm, z_haut_mm, color='yellow', alpha=0.15, label=f'Couche {i}')
        axes[1].set_title(f'Couche {i} — XZ profil (CT complet)')
        axes[1].set_xlabel('X (mm)')
        axes[1].set_ylabel('Z (mm)')
        if pts_g.shape[0] > 0:
            sc1 = axes[1].scatter(pts_g[:, 0], pts_g[:, 2] - dz,
                                  c=z_vals, cmap='coolwarm', s=0.3, alpha=0.9,
                                  vmin=z_p1, vmax=z_p99)
            plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04).set_label('Z PLY (mm)', fontsize=9)
        axes[1].legend(fontsize=9)

        # -------- YZ : fond = volume complet, bandeau = chunk courant --------
        axes[2].imshow(proj_yz_full, cmap='gray', origin='lower',
                       extent=[ct_y0, ct_y1, *ext_z],
                       vmin=vmin_g, vmax=vmax_g, aspect='auto')
        axes[2].axhspan(z_bas_mm, z_haut_mm, color='yellow', alpha=0.15, label=f'Couche {i}')
        axes[2].set_title(f'Couche {i} — YZ face (CT complet)')
        axes[2].set_xlabel('Y (mm)')
        axes[2].set_ylabel('Z (mm)')
        if pts_g.shape[0] > 0:
            sc2 = axes[2].scatter(pts_g[:, 1], pts_g[:, 2] - dz,
                                  c=z_vals, cmap='coolwarm', s=0.3, alpha=0.9,
                                  vmin=z_p1, vmax=z_p99)
            plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04).set_label('Z PLY (mm)', fontsize=9)
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        out_path = f"verif_couches/couche_{i:02d}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"  {len(resultats)} images sauvegardées dans le dossier 'verif_couches/'.")


# =============================================================================
# 6. Visualisation 2D de superposition globale
# =============================================================================

def visualiser_2d_global(pts_ply, vol_ct, dx, dy, dz, res_xy, res_z):
    """
    Projections orthogonales (max intensity) du CT + contour Gocator.
    """
    print("  Génération des coupes 2D globales…")
    Nz, Ny, Nx = vol_ct.shape

    # Coordonnées PLY dans l'espace voxel CT
    x_vox = np.clip(((pts_ply[:, 0] - dx) / res_xy).astype(int), 0, Nx - 1)
    y_vox = np.clip(((pts_ply[:, 1] - dy) / res_xy).astype(int), 0, Ny - 1)
    z_vox = np.clip(((pts_ply[:, 2] - dz) / res_z ).astype(int), 0, Nz - 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Superposition Globale : CT (gris) + Gocator (coolwarm)", fontsize=13)

    # --- XY (vue de dessus, projection Z max) ---
    proj_xy = vol_ct.max(axis=0)
    axes[0].imshow(proj_xy, cmap="gray", origin="lower",
                   extent=[0, Nx * res_xy, 0, Ny * res_xy])
    sc0 = axes[0].scatter(pts_ply[:, 0] - dx, pts_ply[:, 1] - dy,
                          c=pts_ply[:, 2], cmap='coolwarm', s=0.3, alpha=0.5)
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04).set_label('Z (mm)')
    axes[0].set_title("Vue XY (dessus)")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")

    # --- XZ (vue latérale, projection Y max) ---
    proj_xz = vol_ct.max(axis=1)
    axes[1].imshow(proj_xz, cmap="gray", origin="lower",
                   extent=[0, Nx * res_xy, 0, Nz * res_z])
    sc1 = axes[1].scatter(pts_ply[:, 0] - dx, pts_ply[:, 2] - dz,
                          c=pts_ply[:, 2], cmap='coolwarm', s=0.3, alpha=0.5)
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04).set_label('Z (mm)')
    axes[1].set_title("Vue XZ (profil)")
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Z (mm)")

    # --- YZ (vue de face, projection X max) ---
    proj_yz = vol_ct.max(axis=2)
    axes[2].imshow(proj_yz, cmap="gray", origin="lower",
                   extent=[0, Ny * res_xy, 0, Nz * res_z])
    sc2 = axes[2].scatter(pts_ply[:, 1] - dy, pts_ply[:, 2] - dz,
                          c=pts_ply[:, 2], cmap='coolwarm', s=0.3, alpha=0.5)
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04).set_label('Z (mm)')
    axes[2].set_title("Vue YZ (face)")
    axes[2].set_xlabel("Y (mm)")
    axes[2].set_ylabel("Z (mm)")

    plt.tight_layout()
    plt.savefig("verif_alignment_2d.png", dpi=150)
    print("  Sauvegardé : verif_alignment_2d.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cfg    = CONFIG
    res_xy = cfg["RESOLUTION_CT_MM"]

    print("=" * 65)
    print("  PIPELINE ALIGNEMENT GOCATOR <-> MICRO-CT  (ali2.py)")
    print("=" * 65)

    print("\n[1/4] Chargement des couches PLY…")
    pts_ply, nb_couches = charger_couches_ply(cfg)

    print(f"\n[2/4] Chargement CT depuis NPZ…")
    vol_ct, res_z = charger_ct(cfg)

    print("\n[3/4] Calcul de l'alignement (translation TOP-Z)…")
    dx, dy, dz = aligner(pts_ply, vol_ct, res_xy, res_z)

    print("\n[3b/4] Recadrage CT à la taille physique de l'objet…")
    vol_ct_crop, z_off, x_off, y_off = recadrer_ct(
        vol_ct, res_xy, res_z, cfg["TAILLE_OBJET_MM"]
    )
    # Le recadrage décale l'origine Z du tenseur : mettre à jour dz en conséquence
    # Nouveau dz : tel que z_ply = z_vox_crop * res_z + dz_new
    # On avait  : z_ply = z_vox_full * res_z + dz
    # z_vox_crop = z_vox_full - z_off  =>  z_vox_full = z_vox_crop + z_off
    # z_ply = (z_vox_crop + z_off) * res_z + dz = z_vox_crop * res_z + (dz + z_off * res_z)
    dz_crop = dz + z_off * res_z
    # Idem pour dx, dy (décalage XY)
    dx_crop = dx + x_off * res_xy
    dy_crop = dy + y_off * res_xy
    print(f"  Offsets après recadrage : dz_crop={dz_crop:+.3f} mm, "
          f"dx_crop={dx_crop:+.3f} mm, dy_crop={dy_crop:+.3f} mm")

    print("\n[4/4] Extraction et visualisation par couche…")
    resultats = extraire_tranches(
        vol_ct_crop, pts_ply, nb_couches,
        cfg["EPAISSEUR_COUCHE"], res_z, dz_crop
    )
    visualiser_3d(resultats, res_xy, res_z, dx_crop, dy_crop, dz_crop)
    visualiser_superposition_3d_totale(pts_ply, vol_ct_crop, dx_crop, dy_crop, dz_crop, res_xy, res_z)
    visualiser_couches_2d_separees(resultats, vol_ct_crop, res_xy, res_z, dx_crop, dy_crop, dz_crop)
    visualiser_2d_global(pts_ply, vol_ct_crop, dx_crop, dy_crop, dz_crop, res_xy, res_z)

    print("\n[TERMINÉ] Vérifiez verif_couches_3d.png et verif_alignment_2d.png")


if __name__ == "__main__":
    main()
