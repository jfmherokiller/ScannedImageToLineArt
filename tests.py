# coding: utf-8

# # This notebook Attempts to extract the lineart from orbitry images
#
# Start by Importing needed Assets

# In[151]:
from PIL import Image, ImageFilter, ImageOps
import matplotlib.pylab as plt
import numpy as np

plt.rcParams['figure.figsize'] = [20, 9]
import cv2

# In[152]:


Basephoto = cv2.imread("/Users/jfmmeyers/Google Drive/furart/lockott/photo_2018-03-24_13-54-56.jpg", -1)


# In[153]:


def ConvertImageToArray(ImageData):
    return np.asarray(ImageData)


def DisplayDiffrences(OldImage, NewImage):
    f, axarr = plt.subplots(1, 3)

    axarr[0].imshow(OldImage)
    axarr[0].set_title('Old Image')

    axarr[1].imshow(NewImage)
    axarr[1].set_title('New Image')

    axarr[2].imshow(np.abs(ConvertImageToArray(NewImage) - ConvertImageToArray(OldImage)))
    axarr[2].set_title('Changes')

    plt.suptitle('Diffrences between Changed And Original Image', fontsize=16)

    plt.show()


# In[159]:


# remove shadows from image on all color planes
def RemoveShadows(Img):
    rgb_planes = cv2.split(Img)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        cv2.normalize(diff_img, diff_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(diff_img)

    result_norm = cv2.merge(result_norm_planes)
    return result_norm


# cleanup any scanner artificats
NoShadow = RemoveShadows(Basephoto)

Threshold = 222  # the value has to be adjusted for an image of interest
mask = Image.fromarray(NoShadow).convert("L")
mask = mask.point(lambda i: i < Threshold and 255)
DisplayDiffrences(Image.fromarray(Basephoto).convert("L"), ImageOps.invert(mask))
