import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from anisotropic_filtering import *

from skimage import filters
from skimage.filters.thresholding import _cross_entropy
from skimage.filters import threshold_multiotsu

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


img = Image.open("./data/yes/Y44.jpg").convert('L')
img = np.array(img)
print(img.shape)
img_original = img

show_image(img, title="Image de test")



#Filtrage anistropique pour nettoyage

img_filtered = anisodiff(img)
img_filtered = anisodiff(img_filtered)
img_filtered = anisodiff(img_filtered)
images = [img, img_filtered]
titles = ["Avant filtrage", "Post-Filtrage"]

img = img_filtered


"""
show_images(images, titles)
"""


#SKULL STRIPPING

#1. Thresholding by cross-entropy
thresholds = np.arange(np.min(img) + 1.5, np.max(img) - 1.5)
entropies = [_cross_entropy(img, t) for t in thresholds]

optimal_camera_threshold = thresholds[np.argmin(entropies)]
img_thresholded = img > optimal_camera_threshold


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

print('The brute force optimal threshold is:', optimal_camera_threshold)
print('The computed optimal threshold is:', filters.threshold_li(img))

plt.show()


def find_horizontal_length(img):

    h, w = img.shape
    img_center = (h//2,w//2)
    length_w_1 = 0
    length_w_2 = 0

    for i in range(img_center[1]-1,0,-1):
        if img[img_center[0],i] == 1:
            length_w_1+=1
        else:
            break

    for i in range(img_center[1], w+1):
        if img[img_center[0],i] == 1:
            length_w_2 += 1
        else:
            break
    
    return max(length_w_1,length_w_2)

def find_vertical_length(img):
    h, w = img.shape
    img_center = (h//2,w//2)
    length_h_1 = 0
    length_h_2 = 0
    for i in range(img_center[0]-1,0,-1):
        if img[i,img_center[1]] == 1:
            length_h_1+=1
        else:
            break

    for i in range(img_center[0], h+1):
        if img[i,img_center[1]] == 1:
            length_h_2 += 1
        else:
            break
    
    return max(length_h_1,length_h_2)




#2 Filling in the gaps.
img_thresholded[img_thresholded==True] = 255
img_thresholded[img_thresholded==False] = 0
img_thresholded = np.astype(img_thresholded, np.uint8)

kernel = np.ones((8,8),np.uint8)


#Fermeture et Ouverture pour nettoyer
img_thresholded = cv2.morphologyEx(img_thresholded,cv2.MORPH_OPEN,kernel)

show_image(img_thresholded, title="Post Morphological OP", cmap="gray", show=True)

#Find image Orientation:
def find_best_image_orientation(img_arr):
    vertical_lengths = [ ]
    angles = [i for i in range(0,181)]

    array_to_img_object = Image.fromarray(img_arr)
    for angle in angles:
        current_image = array_to_img_object.rotate(angle)
        img_back_to_array = np.array(current_image)
        vertical_lengths.append(find_vertical_length(img_back_to_array))



    correct_angle = angles[np.argmax(vertical_lengths)]
    print("Correct angle : ", correct_angle)

    img = np.array(array_to_img_object.rotate(correct_angle))

    show_image(img, title="New Angle", cmap="gray", show=True)

    return img


#Ellipse drawing:
img_thresholded = find_best_image_orientation(img_thresholded)
h,w= img_thresholded.shape
center = (w//2,h//2)
axes = (find_horizontal_length(img_thresholded), find_vertical_length(img_thresholded))
angle = 0
startAngle = 0
endAngle = 360
color = (255,255,255)
thickness = -1
 
#ellipse(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> img
img_ellipse = np.zeros_like(img_original)
cv2.ellipse(img_ellipse, center, axes, angle, startAngle, endAngle, color, thickness)

kernel = np.ones((3,3),np.uint8)
#img_ellipse = img_ellipse[:,:,0] #Tous les canaux à 255, donc en prendre 1 revient à une image grayscale
img_ellipse = cv2.dilate(img_ellipse,kernel,iterations = 1)
img_ellipse_mask = img_ellipse > 254

show_image(img_ellipse, title="Skull stripping Mask", cmap="gray", show=True)

img_skull_stripped = img_original*img_ellipse_mask
show_image(img_skull_stripped, title="Original Post Skull Stripping", cmap="gray", show=True)




def multi_otzu(image) : 
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

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

multi_otzu(img_skull_stripped)



#3.Top Hat filtering

#tophat = cv2.morphologyEx(img_skull_stripped, cv2.MORPH_TOPHAT, kernel)