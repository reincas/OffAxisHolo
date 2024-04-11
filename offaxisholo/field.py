##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides functions to generate quadratic field arrays. All
# length parameters are expected as unitless multiples of the wavelength.
#
##########################################################################

import numpy as np


def mesh(N, pitch):
    
    """ Return quadratic x and y position arrays with given pitch and origin
    of the coordinate system at the array center."""

    y, x = np.indices((N, N), dtype=float)
    x -= 0.5*np.max(x)
    y -= 0.5*np.max(y)
    x *= pitch
    y *= pitch
    return y, x


def norm(img):
    
    """ Return given complex field with total magnitude normalized to a value
    of 1.0."""
    
    return img / np.sum(np.abs(img))


def planar(N, theta, phi):
    
    """ Return a planar wave field with magnitude 1.0 and tilted by a polar
    angle theta and an azimutal angle phi as complex (N,N) array. """

    fx = np.sin(theta) * np.cos(phi)
    fy = np.sin(theta) * np.sin(phi)
    y, x = mesh(N, 1.0)
    F = np.exp(2j*np.pi * (fx*x + fy*y))
    return F


def spherical(N, pitch, z, approx=False):
    
    """ Return a spherical wave field with magnitude 1.0 at a distance z from
    the center of the sphere as a complex (N,N) array. The center of the sphere
    is located on the optical axis. """
    
    # FIXME: Check the equation!
    y, x = mesh(N, pitch)
    if approx:
        F = np.exp(2j*np.pi * (z + (x*x + y*y) / (2*z)))
    else:
        F = np.exp(2j*np.pi * np.sqrt(x*x + y*y + z*z))
    return F


def aperture(N, r0, pitch, x0=0, y0=0):
    
    """ Return (N,N) array with value 1.0 inside and 0.0 outside a circular
    aperture with radius r0 and offset (x0,y0) relative to the center of the
    array. """
    
    y, x = mesh(N, pitch)
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    F = np.where(r <= r0, 1.0, 0.0)
    return F


