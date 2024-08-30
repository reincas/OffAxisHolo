##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module contains the class Zernike, which provides Zernike polynoms
# on a rectangular grid. The class structure is inspired by [1]. The
# definition of the circular Zernike polynomials is taken from [2] and the
# Gram-Schmidt orthonormalization is based on [3].
#
# [1] https://github.com/jacopoantonello/zernike
# [2] https://en.wikipedia.org/wiki/Zernike_polynomials
# [3] Robert Upton and Brent Ellerbroek, "Gram–Schmidt orthogonalization of
#     the Zernike polynomials on apertures of arbitrary shape," Opt. Lett. 29,
#     2840-2842 (2004), https://doi.org/10.1364/OL.29.002840
#
##########################################################################

import math
import numpy as np


class Zernike(object):
    
    """ Zernike polynoms on a regular grid. Provides the method eval_grid()
    to calculate a superposition of polynomials and the method fit_grid() for
    a least-squares fit of the polynomials to a given phase map. """
    
    def __init__(self, rank, scheme="ANSI", dtype=float):
        
        """ Initialize certain coefficients for all Zernike polynoms up to
        the given rank in the given indexing scheme Noll, ANSI, or fringe.
        The method make_grid() should be called next to generate the actual
        polynomials for further calculations. """
        
        assert(isinstance(scheme, str))
        assert(isinstance(rank, int))
        assert(rank >= 0)
        
        # Maximum rank
        self.rank = rank
        
        # Numpy data type to use
        self.dtype = dtype
        
        # List of all Zernike (n,m) index pairs up to the given rank
        self.scheme = scheme.lower()
        if self.scheme == "noll":
            self.indices = self._noll_indices(self.rank)
        elif self.scheme == "ansi":
            self.indices = self._ansi_indices(self.rank)
        elif self.scheme == "fringe":
            self.indices = self._fringe_indices(self.rank)
        else:
            raise RuntimeError("Unknown indexing scheme %s!" % scheme)
        self.size = len(self.indices)
        
        # Normalizing factor for each Zernike index pair
        self._norm = []
        for n, m in self.indices:
            if m == 0:
                self._norm.append(math.sqrt(n+1))
            else:
                self._norm.append(math.sqrt(2*(n+1)))

        # Magnitudes and polynomial exponents for each Zernike index pair
        # FIXME: Use cache to utilize poly(n,m) == poly(n,-m)
        self._poly = []
        for n, m in self.indices:
            m = abs(m)
            poly = []
            for s in range((n - m) // 2 + 1):
                mag = (1 - 2*(s % 2))
                mag *= math.factorial(n - s)
                mag /= math.factorial(s)
                mag /= math.factorial((n + m) // 2 - s)
                mag /= math.factorial((n - m) // 2 - s)
                exp = n - 2*s
                poly.append((mag, exp))
            self._poly.append(poly)
        
        # No grid yet
        self.shape = None
        self.fsize = None
        self.Z = None
        self.F = None
        

    def _noll_indices(self, rank):
        
        """ Build list of Zernike (n,m) indices for the given rank in the
        Noll indexing scheme. """
        
        indices = []
        for r in range(rank+1):
            n = r
            for m in range(n % 2, n+1, 2):
                if m == 0:
                    indices.append((n, m))
                else:
                    if len(indices) % 2:
                        indices.append((n, m))
                        indices.append((n, -m))
                    else:
                        indices.append((n, -m))
                        indices.append((n, m))
        return indices


    def _ansi_indices(self, rank):
        
        """ Build list of Zernike (n,m) indices for the given rank in the
        ANSI indexing scheme (OSA scheme). """
        
        indices = []
        for r in range(rank+1):
            n = r
            for m in range(-n, n+1, 2):
                indices.append((n, m))
        return indices


    def _fringe_indices(self, rank):
        
        """ Build list of Zernike (n,m) indices for the given rank in the
        fringe indexing scheme (University of Arizona scheme). """
        
        indices = []
        for r in range(rank+1):
            n = r
            m = n
            while m > 0:
                indices.append((n, m))
                indices.append((n, -m))
                n += 1
                m -= 1
            indices.append((n, m))
        return indices


    def _value(self, j, rho, theta):
        
        """ Value of the Zernike polynomial with index j. """
        
        assert(isinstance(j, int))
        assert(j >= 0 and j < self.size)

        return  self._radial(j, rho) * self._angular(j, theta)

        
    def _radial(self, j, rho):
        
        """ Radial part of the Zernike polynomial with index j. """
        
        assert(isinstance(j, int))
        assert(j >= 0 and j < self.size)

        poly = [mag * rho**exp for mag, exp in self._poly[j]]
        return self._norm[j] * np.sum(poly, axis=0)


    def _angular(self, j, theta):
        
        """ Angular part of the Zernike polynomial with index j. """
        
        assert(isinstance(j, int))
        assert(j >= 0 and j < self.size)

        m = self.indices[j][1]
        if m >= 0:
            return np.cos(m * theta)
        return np.sin(-m * theta)
    
    
    def make_grid(self, shape):
        
        """ Define rectangular grid of given shape with area pi. Determine
        orthonormal polynomials F based on the circular Zernike polynomials
        using the Gram-Schmidt method. """
        
        # Cartesian coordinate system for area with size pi
        h, w = self.shape = shape
        self.fsize = w * h
        p = np.sqrt(np.pi / self.fsize)
        x = np.arange(w) * p
        y = np.arange(h) * p
        x -= 0.5 * x[-1]
        y -= 0.5 * y[-1]
        x, y = np.meshgrid(x, y)
        
        # Polar coordinates of all coordinate points as vectors
        rho = np.sqrt(np.square(x) + np.square(y)).ravel()
        theta = np.arctan2(y, x).ravel()
        
        # Zernike polynomials for all coordinate points
        self.Z = Z = np.zeros((self.fsize, self.size), dtype=self.dtype)
        for j in range(self.size):
            Z[:,j] = self._value(j, rho, theta)

        # Gram–Schmidt orthogonalization of the Zernike polynomials Z results
        # in orthonormal polynomials F on the rectangular grid
        self.F = F = np.zeros(Z.shape, dtype=self.dtype)
        for j  in range(self.size):
            F[:,j] = Z[:,j]
            for i in range(j):
                F[:,j] -= F[:,i] * np.sum(Z[:,j] * F[:,i]) / np.pi
            F[:,j] /= np.sqrt(np.sum(np.square(F[:,j])) / np.pi)        
    
    
    def eval_grid(self, coeff, matrix=False):
        
        """ Return superposition of F poynomials using the given weight
        factors as vector or matrix. """
        
        assert(self.F is not None)
        assert(isinstance(coeff, np.ndarray))
        assert(coeff.shape == (self.size,))

        if coeff.dtype != self.dtype:
            coeff = coeff.astype(self.dtype)
        
        result = np.matmul(self.F, coeff)
        if matrix:
            return result.reshape(self.shape)
        return result
    
    
    def fit_grid(self, phase):
        
        """ Fit F polynomials to the given phase map vector or matrix and
        return the resulting weight factors. """
        
        assert(self.F is not None)
        assert(isinstance(phase, np.ndarray))
        if len(phase.shape) == 1:
            assert(phase.size == self.fsize)
        else:
            assert(phase.shape == self.shape)
            phase = phase.ravel()
            
        if phase.dtype != self.dtype:
            phase = phase.astype(self.dtype)
            
        return np.linalg.lstsq(self.F, phase, rcond=None)[0]


