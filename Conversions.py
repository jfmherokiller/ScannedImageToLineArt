# coding: utf-8

import matplotlib.pylab as plt
import numpy as np
import cv2
import svgwrite

plt.rcParams['figure.figsize'] = [20, 9]

Basephoto = cv2.imread("/Users/jfmmeyers/Google Drive/furart/lockott/photo_2018-03-24_13-54-56.jpg",
                       cv2.IMREAD_GRAYSCALE)


class ExtractLineArt:
    UnprocessedImage = ''

    def __init__(self, imagePath: str):
        self.UnprocessedImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)


def ConvertImageToArray(ImageData):
    return np.asarray(ImageData)


def DisplayDiffrences(OldImage, NewImage, ExtractedLines):
    f, axarr = plt.subplots(1, 4)

    axarr[0].imshow(OldImage)
    axarr[0].set_title('Old Image')

    axarr[1].imshow(NewImage, cmap='binary')
    axarr[1].set_title('New Image')

    axarr[2].imshow(np.abs(NewImage) - OldImage)
    axarr[2].set_title('Changes')

    axarr[3].imshow(ExtractedLines, cmap='binary')
    axarr[3].set_title('Found Line Art')
    plt.suptitle('Diffrences between Changed And Original Image and Found Lines', fontsize=16)

    plt.show()


# In[159]:


# remove shadows from image on all color planes
def RemoveShadows(Img: np.ndarray) -> np.ndarray:
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


def ProcessImage(InputImage: np.ndarray, DesiredThreshold: int):
    # cleanup any scanner artificats
    NoShadow = RemoveShadows(InputImage)
    Lineart = DarkenLines(NoShadow, DesiredThreshold)
    ExtractedPoints = FindLines(Lineart)
    DrawnLines = DisplayLines(Basephoto, ExtractedPoints)
    WriteContoursToSVG(ExtractedPoints, "test.svg", Basephoto)
    DisplayDiffrences(Basephoto, Lineart, DrawnLines)


def DarkenLines(InputImage: np.ndarray, DesiredThreshold: int):
    mask = InputImage
    cv2.threshold(InputImage, DesiredThreshold, 255, cv2.THRESH_BINARY_INV, mask)
    cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY, mask)
    # cv2.bitwise_not(mask, mask)
    return mask


def FindLines(InputImage):
    contours = cv2.findContours(InputImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    return contours


def DisplayLines(OriginalImage: np.ndarray, Contours: np.ndarray) -> np.ndarray:
    height, width = OriginalImage.shape
    NewImage = np.zeros((height, width, 3), np.uint8)
    cv2.drawContours(NewImage, Contours[1], -1, (0, 255, 0), 3)
    return NewImage


def WriteContoursToSVG(Contours: np.ndarray, FileName: str, OriginalImage: np.ndarray):
    height, width = OriginalImage.shape
    dwg = svgwrite.Drawing(FileName, size=(width, height))
    for Contour in Contours[1]:
        PointSet = []
        for PointPiece in Contour.tolist():
            PointSet.append(tuple(PointPiece[0]))
            dwg.add(dwg.polyline(PointSet))
    # save svg here outside of loop
    dwg.save()


def SortContours(Contours: np.ndarray):
    def T(i):
        children = []
        for j, h in enumerate(Contours[1]):
            if h[3] == i:
                children.append((h, j))

        def function1(h):
            return h[0][1]

        children.sort(key=function1)
        return {c[1]: T(c[1]) for c in children}


ProcessImage(Basephoto, 222)
