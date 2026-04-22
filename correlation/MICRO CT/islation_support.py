import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import ConvexHull
from skimage.filters import threshold_multiotsu
import scipy.ndimage as ndimage

def remove_support_by_variance(volume, seuil_variance, patch_size=200):
    """
    Supprime les premières tranches Y contenant le support en utilisant 
    la variance d'une colonne centrale.
    """
    Z, Y, X = volume.shape
    
    # 1. Définir la zone centrale (xmid, zmid)
    z_mid, x_mid = Z // 2, X // 2
    
    # Limites du patch pour ne pas déborder
    half_patch = patch_size // 2
    z_start = max(0, z_mid - half_patch)
    z_end = min(Z, z_mid + half_patch)
    x_start = max(0, x_mid - half_patch)
    x_end = min(X, x_mid + half_patch)
    
    # 2. Calculer la variance pour chaque plan Y, mais uniquement dans le patch central
    variances_y = np.zeros(Y)
    
    for y in range(Y):
        patch = volume[z_start:z_end, y, x_start:x_end]
        variances_y[y] = np.var(patch)
        
    # 3. Trouver le premier index Y où la variance dépasse le seuil
    indices_piece = np.where(variances_y > seuil_variance)[0]
    
    if len(indices_piece) == 0:
        print("Attention: Le seuil de variance n'a jamais été atteint. Renvoie le volume complet.")
        return volume, 0, variances_y
        
    y_cut = indices_piece[0]
    print(f"Transition Support -> Pièce détectée à Y = {y_cut}")
    
    # 4. Couper le volume
    volume_nettoye = volume[:, y_cut:, :]
    
    return volume_nettoye, y_cut, variances_y

def extraction_pièce(volume):
    sample = volume[::10, ::10, ::10]
    try:
        thresh = threshold_multiotsu(sample, classes=3)
        print(f"Seuils multi-Otsu détectés : {thresh[0]:.1f}, {thresh[1]:.1f}")
        mask = volume.copy()
        mask[mask < thresh[1]] = 0
    except ValueError:
        print("Erreur threshold_multiotsu, aucun seuillage appliqué.")
        mask = volume.copy()
    return mask

def convex_hull_mask(volume):
    volume_proj = volume[:, 40:, :] if volume.shape[1] > 40 else volume
    proj_xy = np.sum(volume_proj, axis=2)
    proj_xz = np.sum(volume_proj, axis=1)
    proj_yz = np.sum(volume_proj, axis=0)
    
    def to_uint8(img):
        img_min = min(np.min(proj_xy), np.min(proj_xz), np.min(proj_yz))
        img_max = max(np.max(proj_xy), np.max(proj_xz), np.max(proj_yz))
        if img_max > img_min:
            img_norm = (img - img_min) * 255 / (img_max - img_min)
        else:
            img_norm = img * 0
        return img_norm.astype(np.uint8)

    img_xy = to_uint8(proj_xy)
    img_xz = to_uint8(proj_xz)
    img_yz = to_uint8(proj_yz)
    
    _, thresh_xy = cv2.threshold(img_xy, 70, 255, cv2.THRESH_BINARY)
    _, thresh_xz = cv2.threshold(img_xz, 50, 255, cv2.THRESH_BINARY)
    _, thresh_yz = cv2.threshold(img_yz, 10, 255, cv2.THRESH_BINARY)

    thresh_xy[thresh_xy > 20] = 255
    thresh_xz[thresh_xz > 20] = 255
    thresh_yz[thresh_yz > 20] = 255
    thresh_xy[thresh_xy < 20] = 0
    thresh_xz[thresh_xz < 20] = 0
    thresh_yz[thresh_yz < 20] = 0
    
    def get_hull(thresh_img):
        try:
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                return cv2.convexHull(largest)
        except:
            pass
        return np.array([[[0,0]],[[0,thresh_img.shape[0]]],[[thresh_img.shape[1],thresh_img.shape[0]]],[[thresh_img.shape[1],0]]])

    hull_xy = get_hull(thresh_xy)
    hull_xz = get_hull(thresh_xz)
    hull_yz = get_hull(thresh_yz)

    def approx_hull(hull):
        peri = cv2.arcLength(hull, True)
        eps = 0.001
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        while len(approx) > 8:
            eps += 0.001
            test = cv2.approxPolyDP(hull, eps * peri, True)
            if len(test) < 8:
                break
            approx = test
            if eps > 0.5: break
        return approx

    approx_xy = approx_hull(hull_xy)
    approx_xz = approx_hull(hull_xz)
    approx_yz = approx_hull(hull_yz)

    m_xy = np.zeros_like(thresh_xy)
    m_xz = np.zeros_like(thresh_xz)
    m_yz = np.zeros_like(thresh_yz)
    cv2.drawContours(m_xy, [approx_xy], -1, 255, -1)
    cv2.drawContours(m_xz, [approx_xz], -1, 255, -1)
    cv2.drawContours(m_yz, [approx_yz], -1, 255, -1)

    mask_3d_xy = np.repeat(m_xy[:, :, np.newaxis], volume.shape[2], axis=2)
    mask_3d_xz = np.repeat(m_xz[:, np.newaxis, :], volume.shape[1], axis=1)
    mask_3d_yz = np.repeat(m_yz[np.newaxis, :, :], volume.shape[0], axis=0)

    target_z = max(mask_3d_xy.shape[0], mask_3d_xz.shape[0], mask_3d_yz.shape[0])
    target_y = max(mask_3d_xy.shape[1], mask_3d_xz.shape[1], mask_3d_yz.shape[1])
    target_x = max(mask_3d_xy.shape[2], mask_3d_xz.shape[2], mask_3d_yz.shape[2])
    target_shape = (target_z, target_y, target_x)

    #  np.pad pour uniformiser les tailles des masques 3D avant combinaison
    def fix_shape(m, target):
        paddings = [(0, t - s) for s, t in zip(m.shape, target)]
        return np.pad(m, paddings, mode='constant')

    mask_3d_xy = fix_shape(mask_3d_xy, target_shape)
    mask_3d_xz = fix_shape(mask_3d_xz, target_shape)
    mask_3d_yz = fix_shape(mask_3d_yz, target_shape)

    final_mask_3d = (mask_3d_xy > 0) & (mask_3d_xz > 0) & (mask_3d_yz > 0)
    
    indices = np.argwhere(final_mask_3d)
    if len(indices) == 0:
        return volume
        
    z_min, y_min, x_min = indices.min(axis=0)
    z_max, y_max, x_max = indices.max(axis=0) + 1

    volume_cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_final_cropped = final_mask_3d[z_min:z_max, y_min:y_max, x_min:x_max]
    
    resultat = volume_cropped * mask_final_cropped

    print(f"Rognage 3D : {volume.shape} -> {resultat.shape} (Réduction : {(1 - resultat.size/volume.size)*100:.1f}%)")
    return resultat