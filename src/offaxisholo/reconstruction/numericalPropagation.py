import numpy as np


def angularSpectrum(
    field,
    z,
    wavelength,
    dx,
    dy,
    *,
    n_medium=1.0,
    band_limit=True,
    padding=False,
):
    """
    Numerical propagation of a complex field with the angular spectrum method (ASM).

    Theory
    ------
    The field E(x, y; 0) is decomposed into plane waves by a 2D Fourier transform
    (its angular spectrum A(fx, fy; 0)). Each plane wave propagates along z with the
    longitudinal wave number

        kz(fx, fy) = 2 * pi * sqrt( (n / lambda0)^2 - fx^2 - fy^2 ),

    so that the angular spectrum in the plane z is obtained by multiplication with
    the transfer function of free space

        H(fx, fy; z) = exp(i * kz * z)          for fx^2 + fy^2 <  (n / lambda0)^2
        H(fx, fy; z) = 0                        otherwise (evanescent waves).

    The propagated field is

        E(x, y; z) = IFFT2{ FFT2{E(x, y; 0)} * H(fx, fy; z) }.

    The method is exact within scalar diffraction theory (no paraxial / Fresnel
    approximation). Positive z propagates forward, negative z backward.
    Evanescent components decay exponentially and do not carry information to the
    far field; they are set to zero, which is the standard choice and avoids
    exponential amplification for negative z.

    Band limitation
    ---------------
    For larger distances the sampled transfer function H oscillates faster than the
    frequency sampling interval allows (aliasing of H), which produces artefacts.
    Matsushima and Shimobaba showed that the spectrum has to be limited to

        |fx| < 1 / ( lambda * sqrt( (2 * delta_fx * z)^2 + 1 ) ),
        delta_fx = 1 / (Nx * dx)   (frequency sampling interval),

    and analogously for fy, with lambda = lambda0 / n. For the very short
    propagation distances used in DHM (sub-micrometre to a few micrometre) this
    limit is practically identical to the evanescent cut-off, so band limitation
    is harmless and enabled by default.

    Periodic boundaries and padding
    -------------------------------
    The discrete Fourier transform treats the field as one period of an
    infinitely repeated pattern. Multiplying the spectrum with H is therefore a
    circular (periodic) convolution: light that leaves the window on one side
    re-enters on the opposite side (wrap-around), instead of leaving the field
    of view as it would physically. In addition, if the field values at opposite
    borders do not match (tilt, curvature), the periodic continuation contains a
    jump which diffracts like a sharp edge.

    Padding with zeros removes the wrap-around, but in DHM the field does not
    vanish at the borders (the illumination fills the whole field of view).
    Zero padding then creates an artificial hard aperture at the border whose
    diffraction is just as wrong. Therefore, padding=True works as follows:

        1. c = mean of the field along the window border (background wave),
        2. E = c + (E - c); the constant part c is a plane wave at normal
           incidence and propagates exactly as c * exp(2j*pi*n*z/lambda0),
        3. only (E - c), which is close to zero at the borders, is zero padded
           to twice the size, propagated and cropped.

    Because propagation is linear, this decomposition is exact. It works best
    for aberration compensated fields (flat background).

    Independent of the method: light from outside the recorded field of view,
    which would enter the window during propagation, was never recorded. A
    border band of width ~ |z| * tan(asin(NA / n)) can therefore never be
    reconstructed correctly and should be excluded from evaluation.

    Parameters
    ----------
    field : ndarray
        Complex input field in the plane z = 0.
    z : float
        Propagation distance in metres (negative for backward propagation).
    wavelength : float
        Vacuum wavelength lambda0 in metres.
    dx, dy : float
        Sampling interval (pixel pitch in the object plane) in metres along x
        (columns) and y (rows).
    n_medium : float
        Refractive index of the medium in which the field propagates
        (default 1.0, air). The wavelength in the medium is lambda0 / n.
    band_limit : bool
        Apply the band limitation of Matsushima and Shimobaba (default True).
    padding : bool
        Subtract the border mean (propagated analytically as plane wave) and
        zero pad the remaining field to double size (default False). See above.

    Returns
    -------
    ndarray
        Complex field in the plane z.

    References
    ----------
    J. W. Goodman, "Introduction to Fourier Optics", 3rd ed., Roberts & Company
    (2005), Sec. 3.10 "The angular spectrum of plane waves" (transfer function
    of free space, evanescent waves).

    K. Matsushima and T. Shimobaba, "Band-limited angular spectrum method for
    numerical simulation of free-space propagation in far and near fields,"
    Opt. Express 17, 19662-19673 (2009). https://doi.org/10.1364/OE.17.019662

    U. Schnars and W. Jueptner, "Digital Holography", Springer (2005),
    Sec. 3.2 (numerical reconstruction by the convolution approach / ASM).
    """
    field = np.asarray(field, dtype=complex)
    if field.ndim != 2:
        raise ValueError("2D field required for angular spectrum propagation.")
    if z is None or float(z) == 0.0:
        return field
    z = float(z)

    M, N = field.shape  # rows (y), columns (x)

    # Optional padding against circular wrap-around (see docstring)
    background = 0.0
    if padding:
        border = np.concatenate(
            (field[0, :], field[-1, :], field[1:-1, 0], field[1:-1, -1])
        )
        background = np.mean(border)
        pad_y, pad_x = M // 2, N // 2
        field = np.pad(field - background, ((pad_y, M - pad_y), (pad_x, N - pad_x)))
    Mp, Np = field.shape

    # Spatial frequencies in the standard (unshifted) FFT layout.
    # fftfreq is correct for even and odd array sizes.
    fx = np.fft.fftfreq(Np, d=dx)
    fy = np.fft.fftfreq(Mp, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")

    wavelength_medium = wavelength / n_medium
    argument = (1.0 / wavelength_medium) ** 2 - FX**2 - FY**2
    propagating = argument > 0.0

    kz = 2 * np.pi * np.sqrt(np.where(propagating, argument, 0.0))
    transfer = np.where(propagating, np.exp(1j * kz * z), 0.0)

    if band_limit:
        dfx = 1.0 / (Np * dx)
        dfy = 1.0 / (Mp * dy)
        fx_limit = 1.0 / (wavelength_medium * np.sqrt((2 * dfx * z) ** 2 + 1))
        fy_limit = 1.0 / (wavelength_medium * np.sqrt((2 * dfy * z) ** 2 + 1))
        transfer = transfer * ((np.abs(FX) < fx_limit) & (np.abs(FY) < fy_limit))

    propagated = np.fft.ifft2(np.fft.fft2(field) * transfer)

    if padding:
        propagated = propagated[pad_y : pad_y + M, pad_x : pad_x + N]
        # exact propagation of the plane background wave (normal incidence)
        propagated += background * np.exp(2j * np.pi * z / wavelength_medium)

    return propagated
