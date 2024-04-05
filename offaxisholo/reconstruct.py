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


# =============================================================================
# def locateOrder(holo, size=16):
# 
#     """ Calculate the Fourier spectrum of the given positive real valued
#     hologram image and return the spectral coordinates, the maximum spectral
#     filter radius and the weight of the estimated first diffraction order
#     peak. The global maximum after masking the zero and Nyquist frequencies
#     is taken as first diffraction order. The size parameter is the smoothing
#     radius and thus limits the density of local minima to be considered. The
#     weight of the peak is between 0.0 and 1.0. """
# 
#     # Get spectrum of the real valued image
#     if len(holo.shape) != 2:
#         raise RuntimeError("2D hologram image required!")
#     if np.min(holo) < 0:
#         raise RuntimeError("Real positive hologram image required!")
#     if holo.shape[0] % 2 != 0 or  holo.shape[1] % 2 != 0:
#         raise RuntimeError("Hologram image with even dimensions required!")
#     holo = holo.astype(np.float64)
#     spectrum = np.abs(np.fft.rfft2(holo))
#     h, w = spectrum.shape
# 
#     # Blur and normalize the spectrum
#     blurred = cv.GaussianBlur(spectrum, None, size)
#     #blurred = cv.normalize(blurred, None, 0.0, 1.0, cv.NORM_MINMAX, cv.CV_64F)
#     #### UPDATE doc strings
#     blurred /= blurred[0,0]
# 
#     # Get indices of all local maxima in the spectrum
#     maxmask = (maximum_filter(blurred, size=size) == blurred)
#     points = np.unravel_index(np.nonzero(maxmask.ravel()), maxmask.shape)
#     points = np.concatenate(points, axis=0).T
# 
#     # Strip all local maxima around the zero and the Nyquist frequency. This
#     # strips the dominating zero order peak and many mirror artifacts.
#     s = (w-1) // 2
#     points = [(y, x) for y, x in points if abs(x % (2*s) - s) < s - size//2 and \
#                                            abs(y % (2*s) - s) < s - size//2]
#     if not points:
#         return None, None, 0.0
# 
#     # Take global maximum of the remaining points
#     weights = [blurred[y,x] for y, x in points]
#     y, x = points[np.argmax(weights)]
#     weight = np.max(weights)
# 
#     # Return location and weight of first diffraction order relative to the
#     # zero order. A value y<0 indicates that the first order peak is located
#     # in the upper quadrant.
#     if y >= h//2:
#         y -= h
# 
#     # Done.
#     return x, y, weight
# =============================================================================


# =============================================================================
# def maxRadius(shape, x, y):
# 
#     """ Return maximum radius of the circular mask for the diffraction order
#     filter. The circle must fit into a spectral quadrant. This is determined
#     based on the given shape of a half spectrum originating from
#     np.fft.rfft2(). """
# 
#     # Shape of spectral quadrant
#     h, w = shape
#     h = h // 2
#     w = w - 1
# 
#     # Mirror y coordinate from lower to upper quadrant
#     if y < 0:
#         y += h
# 
#     # Return maximum spectral radius
#     return min(x, w-x, y, h-y)
# 
# =============================================================================

def rollImage(img, x, y):

    """ Roll given image content so that point (x, y) becomes (0, 0). Wrap
    pixels at the image edges. Therefore, no information is lost.  For
    x = w//2 and y = h//2, the function is equivalent to np.fft.fftshift(img).
    """

    return np.roll(img, (-y, -x), axis=(0,1))


# =============================================================================
# def shiftSpectrum(spectrum, x, y, r):
# 
#     """ Take a half spectrum originating from np.fft.rfft2(img) and return
#     a quarter spectrum with the spectral position (x, y) rolled to (0, 0) and
#     masked by a circular mask with the given radius. The spectral position is
#     expected relative to the origin. Thus y<0 selects the upper quadrant. """
# 
#     # Distance between zero and first order
#     r0 = np.sqrt(x*x + y*y)
# 
#     # Maximum spectral radius allowed for given coordinates
#     rmax = maxRadius(spectrum.shape, x, y)
# 
#     # Select the upper or lower quadrantaddressed by (x, y)
#     h, w = spectrum.shape
#     if y < 0:
#         spectrum = spectrum[h//2:,:w-1]
#         y += h // 2
#     else:
#         spectrum = spectrum[:h//2,:w-1]
# 
#     # Determine the maximum radius which fits into a quadrant
#     if r is None:
#         r = rmax
#     elif r >= r0:
#         raise RuntimeError("Spectral radius includes zero order peak!")
#     elif r > rmax:
#         print("*** Warning: spectral radius is too large!")
# 
#     # Roll (x, y) to (0, 0)
#     spectrum = rollImage(spectrum, x, y)
# 
#     # Generate a circular mask with radius r centered at (0, 0) and wrapped
#     # around the image corners
#     h, w = spectrum.shape
#     X = np.arange(w)
#     X = np.where(X < w-x, X, X-w)
#     Y = np.arange(h)
#     Y = np.where(Y < h-y, Y, Y-h)
#     X, Y = np.meshgrid(X, Y)
#     mask = (X*X + Y*Y <= r*r).astype(np.uint8)
# 
#     # Return the masked spectrum
#     return spectrum * mask
# 
# =============================================================================


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

    return np.fft.ifft2(np.fft.fftshift(spectrum))


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
    spectrum = np.fft.fftshift(np.fft.fft2(holo))

    # Roll the given first order coordinates to the centre of the spectrum
    spectrum = rollImage(spectrum, fx, fy)
    
    # Apply circular aperture with radius r
    spectrum = circularMask(spectrum, r)

    # Calculate and return the wave field from the first order spectrum
    field = getField(spectrum)
    return field


# =============================================================================
# def polyTerms(shape, order):
# 
#     """ Return a stack of polynomical pixel coordinate terms up to the given
#     maximum polynomial order and a list of their names. """
# 
#     # Normalize order parameter
#     order = abs(int(order))
# 
#     # Linear terms
#     Y, X = np.indices(shape, dtype=float)
#     X -= 0.5*np.max(X)
#     Y -= 0.5*np.max(Y)
# 
#     # Build polynomial terms in increasing order
#     terms = []
#     names = []
#     for n in range(order+1):
# 
#         # Append n+1 terms for order n
#         for i in range(n+1):
# 
#             # Constant base term (X^0 * Y^0 = 1)
#             term = np.ones(shape, dtype=float)
#             name = ""
# 
#             # Calculate term i (X^(n-i) * Y^i)
#             for j in range(n):
#                 if j < n - i:
#                     term *= X
#                     name += "x"
#                 else:
#                     term *= Y
#                     name += "y"
# 
#             # Append term i
#             terms.append(term)
#             names.append(name)
# 
#     # Return stack of polynomial terms and their names
#     terms = np.stack(terms, axis=2)
#     return terms, names
# 
# 
# def phaseFit(phase, terms):
# 
#     """ Unwrap the given 2D phase array and perform a least-squares fit
#     of the given polynomial terms to it. Return a list of the fitting
#     coefficients for the terms. """
# 
#     # Unwrap the 2D phase array
#     phase = unwrap_phase(phase)
# 
#     # Least-squares fit of 2D quadratic polynomial to the given phase array
#     a = terms.reshape((phase.size, -1))
#     b = phase.ravel()
#     fit_coeff = np.linalg.lstsq(a, b, rcond=None)[0]
# 
#     # Return fitted phase array and fitting coefficients
#     return fit_coeff
# 
# =============================================================================

# =============================================================================
# def refHolo(img, blur=None, order=None, ref=None):
# 
#     """ Take and evaluate a reference image. Determine the spectral
#     location of the first diffraction order and apply a polynomial fit
#     to the phase of the wave field. """
# 
#     if ref is None:
#         ref = SimpleNamespace()
#     ref.img = np.array(img)
#     if blur is not None:
#         ref.blur = blur
#     if order is not None:
#         ref.order = order
# 
#     ref.fx, ref.fy, ref.r, ref.weight = locateOrder(ref.img, ref.blur)
#     field = getField(ref.img, ref.fx, ref.fy, ref.r)
#     phase = np.angle(field)
#     ref.terms, ref.names = polyTerms(phase.shape, ref.order)
#     ref.fit = phaseFit(phase, ref.terms)
#     return ref
# 
# =============================================================================
