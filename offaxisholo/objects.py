##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides functions to generate quadratic object field arrays.
#
##########################################################################

import numpy as np
import cv2 as cv

from . import field
from .usaf1951 import chart
 

def usaf(N, pitch):
    
    """ Return a quadratic 1951 USAF resolution test chart as float (N,N)
    array with a value range of [0..1]. """
    
    F = chart(pitch, N, N).astype(float) / 255
    return F
    
    
def gray_img(N, fn):
    
    """ Read quadratic image from file and return it as float (N,N) array with
    a value range of [0..1]. """
    
    F = cv.imread(fn)
    F = cv.cvtColor(F, cv.COLOR_BGR2GRAY)
    F = cv.normalize(F, None, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_64F)
    if F.shape != (N, N):
        F = cv.resize(F, (N, N))
    return F


def cross(N, w):
    
    """ Return a centered cross with given line width in pixels as quadratic
    float (N,N) array with a value range of [0..1]. """
    
    F = np.zeros((N,N), dtype=float)
    F[:,N//2-w:N//2+w] = 1.0
    F[N//2-w:N//2+w,:] = 1.0
    return F


def pixel(N, x, y):
    
    """ Return a quadratic float (N,N) array with value 1.0 for pixel x,y
    relative to the center and 0.0 everywhere else. """
    
    F = np.zeros((N,N), dtype=float)
    x += N // 2
    y += N // 2
    F[y,x] = 1.0
    return F


def circle(N, r0, w):

    """ Return a centered circle with given radius and line width in pixels
    as quadratic float (N,N) array with a value range of [0..1]. """
    
    y, x = field.mesh(N, 1.0)
    r = np.sqrt(x*x + y*y)
    F = np.where(np.abs(r-r0) < w, 1.0, 0.0)
    return F


def checker(N):
    
    """ Return a single pixel checker board as quadratic float (N,N) array
    with a value range of [0..1]. """
    
    y, x = np.indices((N, N), dtype=int)
    F = np.where((x+y) % 2 > 0, 1.0, 0.0)
    return F


def random(N):
    
    """ Return a random quadratic float (N,N) array with a value range of
    [0..1] as normal distribution. """
    
    rng = np.random.default_rng()
    F = rng.standard_normal((N, N))
    return F


def asphase(F, scale=0.5):
    
    """ Return given a complex phase field with the magnitude 1.0 and the
    phase 2*pi*F*scale. """
    
    F = np.exp(2j*np.pi * F*scale)
    return F
