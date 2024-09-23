def angularSpectrum(self, field, z, wavelength, dx, dy):
    """
    # Function to diffract a complex field using the angular spectrum approximation
    # Inputs:
    # field - complex field
    # z - propagation distance [unit like wavelength]
    # wavelength - wavelength [unit like propagation distance]
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

    spectrum = self.hologram.getSpectrum(field)
    # spectrum = np.fft.fftshift(field)
    # spectrum = np.fft.fft2(spectrum)
    # spectrum = np.fft.fftshift(spectrum)

    phase = np.exp(
        1j * z * np.pi * np.sqrt(np.power(1 / wavelength, 2) - (np.power(X * dfx, 2) + np.power(Y * dfy, 2))))
    tmp = spectrum * phase

    field_prop = self.hologram.getField(tmp)
    # field_prop = np.fft.ifftshift(tmp)
    # field_prop = np.fft.ifft2(field_prop)
    # field_prop = np.fft.ifftshift(field_prop)
    return field_prop
