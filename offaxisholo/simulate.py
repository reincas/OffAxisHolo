##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module contains functions to simulate the generation of an off-axis
# hologram. All length parameters are expected as unitless multiples of the
# wavelength.
#
##########################################################################

import numpy as np

from . import field


def lens(Fin, pin, f, d=None):
    
    """ Transformation of a complex field Fin with given pixel pitch in the
    distance d from a lens with focal length f to the back focal plane. If
    d=None, the field Fin is located in the front focal plane. Return the 
    output field and its pixel pitch. """
    
    assert len(Fin.shape) == 2
    assert Fin.shape[0] == Fin.shape[1]
    
    # Output pitch
    N = Fin.shape[0]
    pout = f / (N * pin)
    
    # Lens transformation using Fourier transform
    Fout = Fin
    Fout = np.fft.fftshift(Fout)
    Fout = np.fft.fft2(Fout)
    Fout = np.fft.fftshift(Fout)
    
    # Quadratic phase factor
    if d is not None:
        v, u = field.mesh(N, f/(N*pin))
        Fout *= np.exp(1j*np.pi/f * (1-d/f) * (u*u+v*v))
    
    # Return output field and pitch
    return Fout, pout


def propagate(Fin, pin, z):
    
    """ Use the angular spectrum method to calculate the complex field at
    the distance z from an input field at distance 0.0. Return the output
    field and its pixel pitch. Input and output pitch are identical for this
    method. """
    
    assert len(Fin.shape) == 2
    assert Fin.shape[0] == Fin.shape[1]
    
    # No propagation
    if float(z) == 0.0:
        return Fin, pin
    
    # Spectral pitch
    N = Fin.shape[0]
    ps = 1.0/(N*pin)
    fy, fx = field.mesh(N, ps)

    # Propagation using the ASM
    Fs = np.fft.fft2(Fin)
    Fs = np.fft.fftshift(Fs)
    Fs *= np.exp(1j*np.pi*z * (2 - fx*fx - fy*fy))
    Fs = np.fft.fftshift(Fs)
    Fout = np.fft.ifft2(Fs)
    
    # Return output field and pitch
    return Fout, pin
