##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module contains functions to reconstruct digital holograms.
#
##########################################################################

#from types import SimpleNamespace

import numpy as np
import cv2 as cv
from scipy.ndimage import maximum_filter
#from skimage.restoration import unwrap_phase


def locateOrder(holo, size=16):

    """ Calculate the Fourier spectrum of the given positive real valued
    hologram image and return the spectral coordinates, the maximum spectral
    filter radius and the weight of the estimated first diffraction order
    peak. The global maximum after masking the zero and Nyquist frequencies
    is taken as first diffraction order. The size parameter is the smoothing
    radius and thus limits the density of local minima to be considered. The
    weight of the peak is between 0.0 and 1.0. """

    assert len(holo.shape) == 2, "2D hologram image required!"
    assert holo.shape[0] == holo.shape[1], "Quadratic hologram image required!"
    assert holo.shape[0] % 2 == 0, "Hologram image with even dimensions required!"
    assert np.min(holo) >= 0.0, "Real positive hologram image required!"

    # Get spectrum of the hologram image
    holo = holo.astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(holo))
    N = spectrum.shape[0]

    # Blur and normalize the right half of the spectrum
    blurred = cv.GaussianBlur(np.abs(spectrum[:,:N//2]), None, size)
    blurred /= blurred[0,0]

    # Get indices of all local maxima in the spectrum
    maxmask = (maximum_filter(blurred, size=size) == blurred)
    points = np.unravel_index(np.nonzero(maxmask.ravel()), maxmask.shape)
    points = np.concatenate(points, axis=0).T

    # Strip all local maxima around the zero and the Nyquist frequency. This
    # strips the dominating zero order peak and many mirror artifacts.
    s = N // 4
    points = [(y, x) for y, x in points if abs(x % (2*s) - s) < s - size//2 and \
                                           abs(y % (2*s) - s) < s - size//2]
    if not points:
        x, y, weight = None, None, 0.0

    else:
        # Take global maximum of the remaining points
        weights = [blurred[y,x] for y, x in points]
        y, x = points[np.argmax(weights)]
        x -= N // 2
        y -= N // 2
        weight = np.max(weights)

    # Done.
    return spectrum, x, y, weight


def rollImage(img, x, y):

    """ Roll given image content so that point (x, y) becomes (0, 0). Wrap
    pixels at the image edges. Therefore, no information is lost.  For
    x = w//2 and y = h//2, the function is equivalent to np.fft.fftshift(img).
    """

    return np.roll(img, (-y, -x), axis=(0,1))


def circularMask(spectrum, r):
    
    """ Apply circular mask with given radius to the centered spectrum. """

    N = spectrum.shape[0]
    y, x = np.indices((N, N), dtype=float)
    x -= N // 2
    y -= N // 2
    r2 = x*x + y*y
    return np.where(r2 <= r*r, spectrum, 0.0)


def getField(spectrum):
    """ Return complex field from centered spectrum. """
    field = np.fft.ifft2(np.fft.fftshift(spectrum))
    return field


def getSpectrum(field):
    FT = np.fft.fft2(field)
    spectrum = np.fft.fftshift(FT)
    return spectrum

def holo2Field(holo, fx, fy, r):

    """ Calculate the wave field from a given positive real valued hologram
    image based on the given spectral position of the first diffraction order
    relative to the zero order. A circular mask with the given radius is
    applied to the Fourier spectrum in order to extract the first order
    spectrum. """

    # Positive, real valued hologram
    if len(holo.shape) != 2:
        raise RuntimeError("2D hologram image required!")
    if np.min(holo) < 0:
        raise RuntimeError("Real positive hologram image required!")
    if holo.shape[0] % 2 != 0 or  holo.shape[1] % 2 != 0:
        raise RuntimeError("Hologram image with even dimensions required!")
    holo = holo.astype(np.float64)

    # Spatial spectrum of the hologram
    tmp = np.fft.fft2(holo)
    spectrum = np.fft.fftshift(tmp)

    # Roll the given first order coordinates to the centre of the spectrum
    spectrum = rollImage(spectrum, fx, fy)
    
    # Apply circular aperture with radius r
    spectrum = circularMask(spectrum, r)

    # Calculate and return the wave field from the first order spectrum
    field = getField(spectrum)
    return field


def angularSpectrum(field, z, wavelength, dx, dy):
    """
    # Function to diffract a complex field using the angular spectrum approximation
    # Inputs:
    # field - complex field
    # z - propagation distance
    # wavelength - wavelength
    # dx,dy - pixel pitch
    """
    field = np.array(field)
    # sanity check
    assert len(field.shape) == 2, "2D hologram image required!"
    assert field.shape[0] == field.shape[1], "Quadratic hologram image required!"
    assert field.shape[0] % 2 == 0, "Hologram image with even dimensions required!"

    M, N = field.shape
    x = np.arange(0, N, 1)  # array x
    y = np.arange(0, M, 1)  # array y
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    spectrum = np.fft.fftshift(field)
    spectrum = np.fft.fft2(spectrum)
    spectrum = np.fft.fftshift(spectrum)

    phase = np.exp2(1j * z * np.pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))

    tmp = spectrum * phase

    field_prop = np.fft.ifftshift(tmp)
    field_prop = np.fft.ifft2(field_prop)
    field_prop = np.fft.ifftshift(field_prop)

    return field_prop