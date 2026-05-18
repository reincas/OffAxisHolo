##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import unittest
import numpy as np
from offaxisholo import Zernike


class TestZernike(unittest.TestCase):

    def test_zernike(self):

        debug = False
        rank = 4
        scheme = "fringe"
        dtype = np.float64
        eps = np.finfo(dtype).resolution
        fit_scale = 47

        z = Zernike(rank, scheme, dtype=dtype)
        #z.make_grid((16, 24))
        z.make_grid((240, 320))

        # Z[0,0] Zernike value test        
        Zmean = 1.0
        value = np.max(np.abs(z.Z[:,0] - Zmean))
        if debug:
            print("Maximum relative Z[0,0] error:", value / eps)
        self.assertTrue(value < eps)

        # F[0,0] orthonormal polynom value test        
        Fmean = np.mean(z.F[:,0])
        value = np.max(np.abs(z.F[:,0] - Fmean))
        if debug:
            print("Maximum relative F[0,0] error:", value / eps)
        self.assertTrue(value < eps)
        
        # Full orthomormality test
        A = np.zeros((z.size, z.size), dtype=z.dtype)
        for i in range(z.size):
            for j in range(z.size):
                A[i,j] = np.sum(z.F[:,i]*z.F[:,j]) / np.pi
                if i == j:
                    A[i,j] -= 1.0
        value = np.max(np.abs(A))
        if debug:
            print("Maximum relative orthonormality error:", value / eps)
        self.assertTrue(value < 5*eps)

        # Phase fitting test
        np.random.seed(832572645)
        c_in = np.random.random_sample(z.size) * fit_scale
        phase = z.eval_grid(c_in)
        c_out = z.fit_grid(phase)
        value = np.max(np.abs(c_in - c_out))
        if debug:
            print("Maximum relative fitting error:", value / eps)
        self.assertTrue(value < 2000*eps)


if __name__ == '__main__':
    unittest.main()
