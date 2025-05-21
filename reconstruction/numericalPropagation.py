import numpy as np
from numpy.lib.scimath import sqrt

def angularSpectrum(field, z, wavelength, dx, dy):
    """
    Angular spectrum propagation supporting both positive and negative distances
    """
    M, N = field.shape
    x = np.arange(0, N)
    y = np.arange(0, M)
    X, Y = np.meshgrid(x - (N / 2), y - (M / 2), indexing='xy')

    dfx = 1 / (dx * M)
    dfy = 1 / (dy * N)

    # Calculate frequency components
    fx = X * dfx
    fy = Y * dfy

    # Wave number
    k = 2 * np.pi / wavelength

    # Transfer function phase
    kz = sqrt(k ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2 + 0j)

    # Handle evanescent waves
    kz = np.real(kz) + 1j * np.abs(np.imag(kz))

    # Fourier transform and apply propagator
    field_spec = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field)))
    field_spec *= np.exp(1j * kz * z)

    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(field_spec)))
