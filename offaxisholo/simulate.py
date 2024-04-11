##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module contains some functions and the class HoloMicroscope to simulate
# the generation of hologram images from off-axis digital holographic
# microscope. All length parameters are expected as unitless multiples of the
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
        return Fin
    
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
    
    # Return output field
    return Fout


class HoloMicroscope(object):
    
    """ Simulation class for an off-axis digital holographic microscope. """
    
    _params = None
    
    def __init__(self, **params):
        
        """ Initialize the DHM. """
    
        # Initialize parameter dictionary
        self._params = {}
        
        # Either a quadratic illumination field or the field size must be
        # specified. The field size must be a power of two. None may be given
        # for a plane wave illumination. The illumination field will be
        # normalized.
        self.fieldSize = params.get("fieldShape", None)
        self.illuminationField = params.get("illuminationField", None)
    
        # Focal length of microscope objective in wavelength units
        self.objectiveFocalLength = params.get("objectiveFocalLength", None)
        
        # Either the numerical aperture or the pupil radius of the microscope
        # objective must be given. The latter value superseeds the former one.
        self.numericalAperture = params.get("numericalAperture", None)
        self.pupilRadius = params.get("pupilRadius", self.pupilRadius)
    
        # Focal length of tube lens in wavelength units
        self.tubeFocalLength = params.get("tubeFocalLength", None)
        
        # Distance of tube lens from back focal plane of objective in
        # wavelength units. Default is focal length of tube lens, which results
        # in a telecentric configuration.
        self.tubeDistance = params.get("tubeDistance", self.tubeFocalLength)
        
        # Distance of camera sensor from back focal plane of tube lens in
        # wavelength units. Default is zero.
        self.sensorDistance = params.get("sensorDistance", 0.0)
        
        # Either the spectral pixel position or the polar ans azimuth angles
        # of the reference beam must be given.
        self.referencePosition = params.get("referencePosition", None)
        self.referenceTilt = params.get("referenceTilt", self.referenceTilt)
        
        # Magnitude of the reference beam. Default is 1.0.
        self.referenceMagnitude = params.get("referenceMagnitude", 1.0)
        
    
    @property
    def shape(self):
        
        assert self.illuminationField is not None, "Illumination field is required!"
        return self.illuminationField.shape
    
    
    @property
    def fieldSize(self):
        
        assert self.illuminationField is not None, "Illumination field is required!"
        return self.illuminationField.shape[0]
    
    
    @fieldSize.setter
    def fieldSize(self, N):
        
        if N is None:
            return
        assert self.illuminationField is None, "Field size is already fixed!"
        assert isinstance(N, int), "Field size must be an integer!"
        assert N > 0, "Field size must be a positive integer!"
        assert N & (N-1) == 0, "Field size must be a power of two!"
        self.illuminationField = np.ones((N, N), dtype=complex)
        
        
    @property
    def illuminationField(self):
        
        return self._params.get("illuminationField", None)
    
    
    @illuminationField.setter
    def illuminationField(self, Fin):
        
        assert Fin is not None or self.illuminationField is not None, "Unknown illumination field!"
        if Fin is None:
            return
        assert isinstance(Fin, np.ndarray), "Field array required!"
        assert np.issubdtype(Fin.dtype, np.number), "Numeric field array required!"
        assert len(Fin.shape) == 2, "Two-dimensional field required!"
        h, w = Fin.shape
        assert h == w, "Quadratic field required"
        assert w & (w-1) == 0, "Field size must be a power of two!"
        Fin = Fin.astype(complex)
        self._params["illuminationField"] = field.norm(Fin)
        
    
    @property
    def objectiveFocalLength(self):
        
        return self._params.get("objectiveFocalLength", None)
    
    
    @objectiveFocalLength.setter
    def objectiveFocalLength(self, fmo):
        
        assert fmo is not None, "Focal length of objective is required!"
        fmo = float(fmo)
        assert fmo > 0.0, "Focal length of objective must be positive!"
        self._params["objectiveFocalLength"] = fmo


    @property
    def numericalAperture(self):
        
        assert self.objectiveFocalLength is not None, "Focal length of objective is required!"
        assert self.pupilRadius is not None, "Pupil radius of objective is required!"
        return self.pupilRadius / self.objectiveFocalLength
    
    
    @numericalAperture.setter
    def numericalAperture(self, na):
        
        if na is None:
            return
        assert self.objectiveFocalLength is not None, "Focal length of objective is required!"
        na = float(na)
        assert na > 0.0, "NA must be positive!"
        assert na < 1.0, "NA must be less than 1.0!"
        self.pupilRadius = self.objectiveFocalLength * na


    @property
    def pupilRadius(self):
        
        return self._params.get("pupilRadius", None)
    
    
    @pupilRadius.setter
    def pupilRadius(self, ra):
        
        assert ra is not None, "Pupil radius of objective is required!"
        ra = float(ra)
        assert ra > 0.0, "Pupil radius of objective must be positive!"
        self._params["pupilRadius"] = ra


    @property
    def tubeFocalLength(self):
        
        return self._params.get("tubeFocalLength", None)
    
    
    @tubeFocalLength.setter
    def tubeFocalLength(self, ftl):
        
        assert ftl is not None, "Focal length of tube lens is required!"
        ftl = float(ftl)
        assert ftl > 0.0, "Focal length of tube lens must be positive!"
        self._params["tubeFocalLength"] = ftl


    @property
    def tubeDistance(self):
        
        return self._params.get("tubeDistance", None)
    
    
    @tubeDistance.setter
    def tubeDistance(self, dtl):
        
        assert dtl is not None, "Distance of tube lens is required!"
        dtl = float(dtl)
        self._params["tubeDistance"] = dtl


    @property
    def sensorDistance(self):
        
        return self._params.get("sensorDistance", None)
    
    
    @sensorDistance.setter
    def sensorDistance(self, ds):
        
        assert ds is not None, "Distance of camera sensor is required!"
        ds = float(ds)
        self._params["sensorDistance"] = ds


    @property
    def referencePosition(self):
        
        assert self.fieldSize is not None, "Illumination field is required!"
        assert self.referenceTilt is not None, "Tilt of reference wave is required!"
        N = self.fieldSize
        theta, phi = self.referenceTilt
        sinp = np.sin(phi)
        cosp = np.cos(phi)
        if abs(cosp) > abs(sinp):
            fx = N*np.sin(theta) / np.sqrt(1 + (sinp/cosp)**2)
            if cosp < 0.0:
                fx = -fx
            fy = fx * sinp / cosp
        else:
            fy = N*np.sin(theta) / np.sqrt(1 + (cosp/sinp)**2)
            if sinp < 0.0:
                fy = -fy
            fx = fy * cosp / sinp
        return fx, fy


    @referencePosition.setter
    def referencePosition(self, pos):
        
        if pos is None:
            return
        if isinstance(pos, tuple):
            assert len(pos) == 2, "Reference position must be a tuple of two!"
            fx, fy = pos
        else:
            fx = fy = pos
        assert self.fieldSize is not None, "Illumination field is required!"
        N = self.fieldSize
        theta = np.arcsin(np.sqrt(fx*fx + fy*fy) / N)
        phi = np.arctan2(fy, fx)
        self.referenceTilt = theta, phi


    @property
    def referenceTilt(self):
        
        return self._params.get("referenceTilt", None)
    
    
    @referenceTilt.setter
    def referenceTilt(self, tilt):
        
        assert tilt is not None, "Tilt angles of reference wave are required!"
        assert isinstance(tilt, tuple), "Tuple of reference tilt angles required!"
        assert len(tilt) == 2, "Reference tilt must be a tuple of two angles!"
        theta, phi = tilt
        self._params["referenceTilt"] = float(theta), float(phi)


    @property
    def referenceMagnitude(self):
        
        return self._params.get("referenceMagnitude", None)
    
    
    @referenceMagnitude.setter
    def referenceMagnitude(self, mr):
        
        assert mr is not None, "Magnitude of reference wave is is required!"
        mr = float(mr)
        assert mr > 0.0, "Reference magnitude must be positive!"
        self._params["referenceMagnitude"] = mr


    def hologram(self, Fo, po, fields=False):
        
        """ Return hologram with pitch for given complex object transmission
        array with pitch in wavelength units. Return also all fields and their
        pitches if fields is True. """
        
        assert isinstance(Fo, np.ndarray), "Object field array required!"
        assert np.issubdtype(Fo.dtype, np.number), "Numeric field array required!"
        assert Fo.shape == self.shape, "Wrong field shape!"

        assert po is not None, "Object pitch is required!"
        po = float(po)
        assert po > 0.0, "Object pitch must be positive!"

        # Illumination field
        Fp = self.illuminationField
        
        # Object field
        Fo = Fo * Fp
        
        # Aperture plane field (back focal plane of objective)
        fmo = self.objectiveFocalLength
        Fs, ps = lens(Fo, po, fmo)
        N = self.fieldSize
        ra = self.pupilRadius
        Fs *= field.aperture(N, ra, ps)
        
        # Image field (back focal plane of tube lens)
        ftl = self.tubeFocalLength
        dtl = self.tubeDistance
        Fi, pi = lens(Fs, ps, ftl, dtl)
        
        # Reference wave        
        theta, phi = self.referenceTilt
        mr = self.referenceMagnitude
        Fr = mr * field.planar(N, theta, phi)

        # Field on camera sensor with reference wave
        ds = self.sensorDistance
        Fc = propagate(Fi, pi, ds)
        Fc += Fr
        
        # Return hologram image with pitch
        holo = np.abs(Fc)
        if not fields:
            return holo, pi

        # Return hologram image with pitch and all fields with pitches
        fields = {
            "illumination": (Fp, po),
            "object": (Fo, po),
            "pupil": (Fs, ps),
            "image": (Fi, pi),
            "reference": (Fr, pi),
            "sensor": (Fc, pi),
            }
        return holo, pi, fields
