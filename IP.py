import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from anisotropic_filtering import *
from skimage.filters.thresholding import _cross_entropy
from skimage.filters import threshold_multiotsu

from scipy import ndimage


#BROUILLON

def show_image(arr, title="No Title", cmap="viridis", show=False):
    """
    Fonction show_image personelle pour faciliter l'affichage sans écrire des plt. à chaque fois.
    """
    if show == True:
        plt.figure()
        plt.title(title)
        plt.imshow(arr, cmap)
        plt.show()

def show_images(images, titles, is_bgr=False):
    """
    Fonction show_images, qui permet d'afficher plusieurs images sous la forme d'un subplot.
    """
    plt.figure(figsize=(18, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        
        if is_bgr:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
            else:
                plt.imshow(img, cmap='gray')
        else:
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)

        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()





#SKULL STRIPPING

#1. Thresholding by cross-entropy

def cross_entropy_thresholding(img_arr, show = False):
    thresholds = np.arange(np.min(img_arr) + 1.5, np.max(img_arr) - 1.5)
    entropies = [_cross_entropy(img_arr, t) for t in thresholds]

    optimal_camera_threshold = thresholds[np.argmin(entropies)]
    img_thresholded = img_arr > optimal_camera_threshold

    if show == True:
        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('image')
        ax[0].set_axis_off()


        ax[1].imshow(img_thresholded, cmap='gray')
        ax[1].set_title('thresholded')
        ax[1].set_axis_off()

        ax[2].plot(thresholds, entropies)
        ax[2].set_xlabel('thresholds')
        ax[2].set_ylabel('cross-entropy')
        ax[2].vlines(
            optimal_camera_threshold,
            ymin=np.min(entropies) - 0.05 * np.ptp(entropies),
            ymax=np.max(entropies) - 0.05 * np.ptp(entropies),
        )
        ax[2].set_title('optimal threshold')

        fig.tight_layout()

        plt.show()

    return img_thresholded


def make_arr_Int(arr):
    arr_copy = arr
    arr_copy[arr_copy==True] = 255
    arr_copy[arr_copy==False] = 0
    arr_copy = np.astype(arr_copy, np.uint8)

    return arr_copy

def fit_and_shrink_ellipse_mask(mask, shrink_percent=15):

    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No Contours")
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        print("Contours too small")
        return None

    (xc, yc), (major, minor), angle = cv2.fitEllipse(contour)

    h, w = mask.shape
    rotation_matrix = cv2.getRotationMatrix2D((xc, yc), angle, 1.0)
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    new_center = np.dot(rotation_matrix, np.array([xc, yc, 1]))
    new_xc, new_yc = new_center[:2]

    scale = (100 - shrink_percent) / 100.0
    shrunk_major = major * scale
    shrunk_minor = minor * scale

    ellipse_mask = np.zeros_like(rotated_mask, dtype=np.uint8)
    cv2.ellipse(
        ellipse_mask,
        (int(new_xc), int(new_yc)),
        (int(shrunk_major / 2), int(shrunk_minor / 2)),
        0, 0, 360, 1, -1
    )

    return ellipse_mask


def skull_stripping(img_arr, show=False):
    img_skull = img_arr

    show_image(img_skull, title="Skull", cmap="gray", show = show)
    img_skull = ndimage.binary_fill_holes(img_skull)
    img_skull = make_arr_Int(img_skull)
    

    return fit_and_shrink_ellipse_mask(img_skull)






def multi_otzu(image, show=False) : 
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)
    if show == True:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

        # Plotting the original image.
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title('Original')
        ax[0].axis('off')

        # Plotting the histogram and the two thresholds obtained from
        # multi-Otsu.
        ax[1].hist(image.ravel(), bins=255)
        ax[1].set_title('Histogram')
        for thresh in thresholds:
            ax[1].axvline(thresh, color='r')

        # Plotting the Multi Otsu result.
        ax[2].imshow(regions, cmap='jet')
        ax[2].set_title('Multi-Otsu result')
        ax[2].axis('off')

        plt.subplots_adjust()

        plt.show()

    return regions




#########################################################################################
##########################################MAIN###########################################
#########################################################################################
REP_OUI = "data/yes"
REP_NON = "data/no"

def image_to_arr(path) : 
    with Image.open(path) as im : 
        im = im.convert("L")
        arr = np.array(im)
    return arr

def data() :
    fichiers = os.listdir(REP_OUI)
    tumeur = []
    for f in fichiers : 
            if f.endswith(".jpg") : 
                tumeur.append(image_to_arr(REP_OUI + "/" + f))
    fichiers = os.listdir(REP_NON)

    no_tum = []
    for f in fichiers : 
            if f.endswith(".jpg") : 
                no_tum.append(image_to_arr(REP_NON + "/" + f))
    return tumeur, no_tum


def cout(image) : 
    row, col = image.shape
    droite = image[:, :col // 2]
    droite = np.reshape(droite, droite.shape[0] * droite.shape[1])
    gauche = image[:, col // 2 : col//2 * 2]
    gauche = gauche[:, ::-1]
    gauche = np.reshape(gauche, gauche.shape[0] * gauche.shape[1])

    dist = np.sqrt((gauche - droite) ** 2)
    return np.mean(dist)


tumeur, no_tum = data()

couts = []
for image in no_tum:
    img = image

    #Anisotropic Filtering
    img_filtered = anisodiff(img)
    img_filtered = anisodiff(img_filtered)
    img_filtered = anisodiff(img_filtered)


    #Cross entropy-thresholding
    img_thresholded = cross_entropy_thresholding(img_filtered, show=False)
    img_thresholded = make_arr_Int(img_thresholded)

    #Skull Stripping
    img_skull_stripped = skull_stripping(img_thresholded, show=False)
    if img_skull_stripped is None:
        continue
    show_image(img_skull_stripped, title="Skull Stripped Mask", cmap="gray", show=False)

    masked = img * (img_skull_stripped > 0)
    masked_display = cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX)
    masked_display = masked_display.astype(np.uint8)

    show_image(masked_display, title="Skull Stripped", cmap="gray", show=False)

    #Otzu
    img_otzu = multi_otzu(masked_display)
    couts.append(cout(img_otzu))


plt.hist(couts)
plt.show()