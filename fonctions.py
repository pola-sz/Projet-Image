import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from anisotropic_filtering import *
import random
from skimage import filters
from skimage.filters.thresholding import _cross_entropy
from skimage.filters import threshold_multiotsu
from sklearn.neighbors import KernelDensity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import scipy.ndimage as ndimage
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


REP_OUI = "data/yes"
REP_NON = "data/no"


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



def image_to_arr(path) : 
    with Image.open(path) as im : 
        im = im.convert("L")
        arr = np.array(im)
    return arr


def cross_entropy_thresholding(img_arr, show = False):
    thresholds = np.arange(np.min(img_arr) + 1.5, np.max(img_arr) - 1.5)
    entropies = [_cross_entropy(img_arr, t) for t in thresholds]

    optimal_camera_threshold = thresholds[np.argmin(entropies)]
    img_thresholded = img_arr > optimal_camera_threshold

    if show : 
        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        ax[0].imshow(img_arr, cmap='gray')
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

    return arr_copy.astype(int)


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

    
    img_skull = ndimage.binary_fill_holes(img_arr)
    img_skull = make_arr_Int(img_skull)
    if show : 
        _, ax = plt.subplots(1, 2)
        ax[0].imshow(img_arr, cmap = "grey")
        ax[0].set_title("Image avant \ntransformation \nmorphologique")
        ax[1].imshow(img_skull, cmap = "grey")
        ax[1].set_title("Image après \ntransformation \nmorphologique")
        plt.show()
        
    
    return fit_and_shrink_ellipse_mask(img_skull)


def multi_otsu(image, show = False) : 
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    try : 
        thresholds = threshold_multiotsu(image)
    except ValueError : 
        pass

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)
    
    if show : 
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


def symetrie(image) :

    _, col = image.shape
    droite = image[:, :col // 2]
    droite = np.reshape(droite, droite.shape[0] * droite.shape[1])
    histD, _ = np.histogram(droite, np.arange(3))

    gauche = image[:, col // 2 : col//2 * 2]
    gauche = np.reshape(gauche, gauche.shape[0] * gauche.shape[1])
    histG, _ = np.histogram(gauche, np.arange(3))

    distance = np.sqrt( (histD - histG) ** 2)
    return np.mean(distance)


def cout(image) : 
    tranche = image.shape[0] // 10
    cout = []
    for i in range(10) : 
        cout.append(symetrie(image[i * tranche: (i+1) * tranche, :]))

    return np.max(cout)


taille = -1

def nouv_taille(tum, non_tum) : 
        global taille
        list = []
        min_taille = 10000000000000000
        nb_non_tum = 0
        for el in non_tum : 
            if isinstance(el.shape, tuple) and (len(el.shape) == 2) and (np.unique(el).shape[0] > 2):
                list.append(el)
                min_taille = min (min_taille, el.shape[0] * el.shape[1])
                nb_non_tum += 1

        for el in tum : 
            if isinstance(el.shape, tuple) and (len(el.shape) == 2) and (np.unique(el).shape[0] > 2):
                list.append(el)
                min_taille = min (min_taille, el.shape[0] * el.shape[1])
        
        print(taille)
        if taille == -1 : 
            taille = min_taille
        else : 
            min_taille = taille
        print(taille)
        x = np.zeros((len(list), min_taille))
        for i, el in enumerate(list) : 
            el = np.resize(el, (min_taille,))
            x[i] = el
        
        y = np.zeros((len(list), ))
        print(y.shape)
        print(nb_non_tum)
        y[ nb_non_tum : ] = 1

        
        return x, y

def data_from_rep() :
    fichiers = os.listdir(REP_OUI)
    tumeur = []
    for f in fichiers : 
            
            if f.endswith(".jpg") and f != "Y103.jpg" : 
                tumeur.append(image_to_arr(REP_OUI + "/" + f))
    fichiers = os.listdir(REP_NON)
    no_tum = []
    for f in fichiers : 
            if f.endswith(".jpg") : 
                no_tum.append(image_to_arr(REP_NON + "/" + f))
    return tumeur, no_tum


def extraction(base, show = False) : 
    im = []
    for image in base:
        img = image
        #Anisotropic Filtering
        img_filtered = anisodiff(img)
        img_filtered = anisodiff(img_filtered)
        img_filtered = anisodiff(img_filtered)


        #Cross entropy-thresholding
        img_thresholded = cross_entropy_thresholding(img_filtered, show)
        img_thresholded[img_thresholded==True] = 255
        img_thresholded[img_thresholded==False] = 0
        img_thresholded = img_thresholded.astype(np.uint8)

        #Skull Stripping
        img_mask = skull_stripping(img_thresholded)
        img_stripped = img_filtered
        img_stripped[img_mask == 0] = 0

        #Otsu
        img_otsu = multi_otsu(img_stripped, show)
        im.append(img_otsu)
        #couts.append(cout(img_otzu))

    return im


def train_val_split(tumeur, non_tumeur) : 
    random_seed = 145
    split_tum1 = int(0.5 * len(tumeur)) 
    split_tum2 = int(0.75 * len(tumeur)) 
    split_no_tum1 = int(0.5 * len(non_tumeur)) 
    split_no_tum2 = int(0.75 * len(non_tumeur)) 

    tum_train = tumeur[ : split_tum1]
    tum_test = tumeur[split_tum1 : split_tum2 ]
    tum_valid = tumeur[split_tum2 :]

    non_tum_train = non_tumeur[ : split_no_tum1]
    non_tum_test = non_tumeur[split_no_tum1 : split_no_tum2]
    non_tum_valid = non_tumeur[split_no_tum2 : ]


    return tum_train, tum_test, tum_valid, non_tum_train, non_tum_test, non_tum_valid


def kdes(tum, non_tum, bandwidth) : 
    X_plot = np.linspace(0, 10000, 1000)

    kde1 = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(tum.reshape(-1, 1))
    dens_tum = np.exp(kde1.score_samples(X_plot.reshape(-1, 1)))

    kde2 = KernelDensity(kernel='gaussian', bandwidth = bandwidth).fit(non_tum.reshape(-1, 1))
    dens_non_tum = np.exp(kde2.score_samples(X_plot.reshape(-1, 1)))

    plt.plot(X_plot, dens_tum, label = "tumeur")
    plt.plot(X_plot, dens_non_tum, label = "pas de tumeur")
    plt.legend()
    plt.title("Estimation de probabilité avec noyau gaussien")
    plt.show()
    return kde1, kde2


def matrice_conf(kde1, kde2, tum_test, non_tum_test) : 
    X_test = np.concatenate((non_tum_test, tum_test), axis = 0)
    y_test = np.concatenate((np.zeros_like(non_tum_test), np.ones_like(tum_test)), axis = 0)
    dens_tum = np.exp(kde1.score_samples(X_test.reshape(-1, 1))).reshape(-1, 1)
    dens_non_tum = np.exp(kde2.score_samples(X_test.reshape(-1, 1))).reshape(-1, 1)

    y_pred = np.argmax(np.concatenate((dens_tum, dens_non_tum), axis = 1), axis = 1)
    print(str(np.round(accuracy_score(y_test, y_pred) *100, 2) ) + " %")
    return(confusion_matrix(y_test, y_pred))
    


def classif(X_train, y_train, n_com = 40, l_rate = "auto") : 
    clf = make_pipeline(StandardScaler(),PCA(n_com), SVC(gamma=l_rate))
    clf.fit(X_train, y_train)
    return clf

def pred(clf, X) : 
    return clf.predict(X)


