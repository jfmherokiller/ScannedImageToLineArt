# coding: utf-8

import matplotlib.pylab as plt
import numpy as np
import cv2

plt.rcParams['figure.figsize'] = [20, 9]

Basephoto = cv2.imread("/Users/jfmmeyers/Google Drive/furart/lockott/photo_2018-03-24_13-54-56.jpg",
                       cv2.IMREAD_GRAYSCALE)


class ExtractLineArt:
    UnprocessedImage = ''

    def __init__(self, imagePath):
        self.UnprocessedImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)


def ConvertImageToArray(ImageData):
    return np.asarray(ImageData)


def DisplayDiffrences(OldImage, NewImage):
    f, axarr = plt.subplots(1, 3)

    axarr[0].imshow(OldImage)
    axarr[0].set_title('Old Image')

    axarr[1].imshow(NewImage, cmap='binary')
    axarr[1].set_title('New Image')

    axarr[2].imshow(np.abs(NewImage) - OldImage)
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


def ProcessImage(InputImage, DesiredThreshold):
    # cleanup any scanner artificats
    NoShadow = RemoveShadows(InputImage)
    Lineart = DarkenLines(NoShadow, DesiredThreshold)
    ExtractedPoints = FindLines(Lineart)
    print(ExtractedPoints)
    DisplayDiffrences(Basephoto, Lineart)


def DarkenLines(InputImage, DesiredThreshold):
    mask = InputImage
    cv2.threshold(InputImage, DesiredThreshold, 255, cv2.THRESH_BINARY_INV, mask)
    cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY, mask)
    # cv2.bitwise_not(mask, mask)
    return mask


def FindLines(InputImage):
    contours = cv2.findContours(InputImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

    return contours


ProcessImage(Basephoto, 222)
