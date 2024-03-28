##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module contains convenience functions for image processing.
#
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
#from skimage.registration import phase_cross_correlation

# BGR colors
CV_WHITE = (255, 255, 255)
CV_BLUE = (255, 0, 0)
CV_GREEN = (0, 255, 0)
CV_RED = (0, 0, 255)


def read(fn):

    """ Rear image from file and eventually convert it to grayscale. """

    img = cv.imread(str(fn))
    img = gray(img)
    return img


def write(fn, img):

    """ Write image to file. """

    cv.imwrite(str(fn), img)
    

def gray(img):

    """ Eventually convert image to grayscale. """

    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def normcolor(img, norm=True, cmap=None):

    """ Normalize given floating point image and convert it to 8-bit BGR
    image. """

    if cmap is None:
        cmap = "viridis"
    if norm:
        img = cv.normalize(img, None, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_64F)
    img = plt.get_cmap(cmap)(img)[:,:,:3]
    img *= 255
    img = img.astype(np.uint8)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return img


def blur(img, r):

    """ Apply a Gaussian blur to the given image. """

    if img.dtype != float:
        img = img.astype(float)
    if r:
        img = cv.GaussianBlur(img, (0,0), r, borderType=cv.BORDER_DEFAULT)
    return img


def drawCircle(img, x, y, r, color, thickness=1, center=True):
    
    """ Draw circle with given radius and line thickness at given position. The
    position x,y is relative to the image center for center=True and relative
    to the upper left corner otherwise. """

    if center:
        h, w, _ = img.shape
        x = round(x + w/2)
        y = round(y + h/2)
    r = round(r)
    img = cv.circle(img, (x, y), r, color, thickness)
    return img

