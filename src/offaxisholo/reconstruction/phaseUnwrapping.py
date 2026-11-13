import math
import os

from .. import backend

# scikit-image and matplotlib are imported inside the functions that need them,
# so that the GPU capable 'Least Squares' method can be used without them
# (lightweight import for the realtime kernel).

# from kamui import unwrap_dimensional


def phase_unwrapping_fast2d(phase_wrapped):
    """
    Two-dimensional phase unwrapping (scikit-image implementation).

    The wrapped phase (values in (-pi, pi]) is unwrapped by adding integer
    multiples of 2*pi. Pixels are processed in order of their reliability (based
    on the second differences of the phase), following a non-continuous path.
    This makes the algorithm robust against noise and phase residues.

    Reference:
        Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and Munther A.
        Gdeisat, "Fast two-dimensional phase-unwrapping algorithm based on sorting by
        reliability following a noncontinuous path,"
        Appl. Opt. 41, 7437-7444 (2002)
        https://doi.org/10.1364/AO.41.007437

    Backend: scikit-image only runs on the CPU. GPU arrays are copied to the
    host, unwrapped and copied back (same backend and dtype as the input).
    """
    from skimage.restoration import unwrap_phase

    phase_unwrapped = unwrap_phase(backend.to_numpy(phase_wrapped))
    return backend.asarray(
        phase_unwrapped, dtype=phase_wrapped.dtype, like=phase_wrapped
    )


def _wrap(phase):
    """Wrap phase values to [-pi, pi)."""
    xp = backend.get_namespace(phase)
    return xp.remainder(phase + math.pi, 2 * math.pi) - math.pi


def phase_unwrapping_ls_dct(phase_wrapped, congruent=False):
    """
    Unweighted least squares phase unwrapping with the discrete cosine transform.

    NOTE: Not yet validated for DHM data and therefore not the default method.
    It has to be compared with phase_unwrapping_fast2d before it is used for
    quantitative evaluations (see ToDos.txt).

    The unwrapped phase phi is the function whose discrete gradient best fits
    (least squares) the wrapped gradient of the measured phase psi:

        dx_ij = W(psi_i,j+1 - psi_ij),   dy_ij = W(psi_i+1,j - psi_ij),

    with the wrapping operator W(.) to [-pi, pi). Minimising
    sum (phi_i,j+1 - phi_ij - dx_ij)^2 + (phi_i+1,j - phi_ij - dy_ij)^2 leads to
    the discrete Poisson equation

        (phi_i+1,j - 2 phi_ij + phi_i-1,j) + (phi_i,j+1 - 2 phi_ij + phi_i,j-1)
            = rho_ij = (dx_ij - dx_i,j-1) + (dy_ij - dy_i-1,j)

    with Neumann boundary conditions. It is diagonalised by the DCT-II:

        phi_hat_kl = rho_hat_kl / (2 cos(pi k / M) + 2 cos(pi l / N) - 4),

    phi_hat_00 = 0 (the phase offset is arbitrary). Only FFT-like operations are
    used, so the method runs efficiently on the GPU.

    Differences to the reliability sorting method (Fast 2D):
        * If the true phase gradient is below pi everywhere and no noise is
          present, both methods give the same result (up to a constant).
        * Phase residues (noise, undersampling, discontinuities) are not
          isolated but distributed over the whole image: the result is smoothed
          and the height of steps can be underestimated.
        * The result is in general not congruent to the wrapped phase, i.e.
          phi - psi is not an integer multiple of 2 pi. With congruent=True the
          result is made congruent: phi_c = phi + W(psi - phi).

    Parameters
    ----------
    phase_wrapped : array
        Wrapped phase in radians (2D).
    congruent : bool
        Make the result congruent to the wrapped phase (default False).

    Returns
    -------
    array
        Unwrapped phase (same backend as the input) with zero mean.

    References:
        Dennis C. Ghiglia and Louis A. Romero, "Robust two-dimensional weighted
        and unweighted phase unwrapping that uses fast transforms and iterative
        methods," J. Opt. Soc. Am. A 11, 107-117 (1994).
        https://doi.org/10.1364/JOSAA.11.000107

        Dennis C. Ghiglia and Mark D. Pritt, "Two-Dimensional Phase Unwrapping:
        Theory, Algorithms, and Software", Wiley (1998), Sec. 5.3.
    """
    xp = backend.get_namespace(phase_wrapped)
    psi = phase_wrapped
    if psi.ndim != 2:
        raise ValueError("2D phase required for phase unwrapping.")
    M, N = psi.shape

    # wrapped phase differences along x (columns) and y (rows)
    dx = _wrap(psi[:, 1:] - psi[:, :-1])  # (M, N-1)
    dy = _wrap(psi[1:, :] - psi[:-1, :])  # (M-1, N)

    # divergence with Neumann boundaries (zero gradient outside of the image)
    zero_col = xp.zeros((M, 1), dtype=psi.dtype)
    zero_row = xp.zeros((1, N), dtype=psi.dtype)
    dx = xp.concat((zero_col, dx, zero_col), axis=1)  # (M, N+1)
    dy = xp.concat((zero_row, dy, zero_row), axis=0)  # (M+1, N)
    rho = (dx[:, 1:] - dx[:, :-1]) + (dy[1:, :] - dy[:-1, :])

    # solve the Poisson equation in the DCT domain
    rho_hat = backend.dctn(rho, type=2, norm="ortho")
    k = xp.arange(M, dtype=psi.dtype)[:, None]
    l = xp.arange(N, dtype=psi.dtype)[None, :]  # noqa: E741
    denominator = 2 * xp.cos(math.pi * k / M) + 2 * xp.cos(math.pi * l / N) - 4
    is_dc = (k == 0) & (l == 0)
    phi_hat = xp.where(is_dc, 0.0, rho_hat / xp.where(is_dc, 1.0, denominator))
    phi = backend.idctn(phi_hat, type=2, norm="ortho")

    if congruent:
        phi = phi + _wrap(psi - phi)
    return phi


def phase_unwrapping_wrap_around(phase_wrapped, plot=False, path=None):
    """
    Same algorithm as phase_unwrapping_fast2d, but with periodic boundary
    conditions (wrap around) along both axes.

    Reference:
        Herraez et al., Appl. Opt. 41, 7437-7444 (2002)
        https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html
    """
    from skimage.restoration import unwrap_phase

    image_unwrapped_wrap_around = unwrap_phase(
        backend.to_numpy(phase_wrapped), wrap_around=(True, True)
    )

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        fig.colorbar(ax.imshow(image_unwrapped_wrap_around), ax=ax)
        ax.set_title("Unwrapped phase with wrap around")
        if path is not None:
            fig.savefig(os.path.join(path, "phase_unwrapped_wrap_around.png"))
            plt.close(fig)
        else:
            plt.show()
    return backend.asarray(
        image_unwrapped_wrap_around, dtype=phase_wrapped.dtype, like=phase_wrapped
    )


# NOTE: Kamui is not compatible with numpy 2.xx -> not necessary as the fast 2D
#       unwrapping algorithm is always used.
'''
def phase_unwrapping_kamui_normal(phase_wrapped, plot=True, path=None):
    """
    https://github.com/yoyolicoris/kamui
    https://ieeexplore.ieee.org/document/673674
    """
    img_unwrapped = unwrap_dimensional(phase_wrapped)
    if plot:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1, ax2 = ax.ravel()
        fig.colorbar(ax1.imshow(phase_wrapped), ax=ax1)
        ax1.set_title('Wrapped phase')
        fig.colorbar(ax2.imshow(img_unwrapped), ax=ax2)
        ax2.set_title('Unwrapped phase (kamui)')
        if path is not None:
            path = os.path.join(path, 'phase_unwrapped_kamui_normal.png')
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
    return img_unwrapped


def phase_unwrapping_kamui_graph(phase_wrapped, plot=True, path=None):
    """
    Based on the same github respository https://github.com/yoyolicoris/kamui

    https://ieeexplore.ieee.org/document/4099386
    """
    img_unwrapped = unwrap_dimensional(phase_wrapped, method='gc')
    if plot:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1, ax2 = ax.ravel()
        fig.colorbar(ax1.imshow(phase_wrapped), ax=ax1)
        ax1.set_title('Wrapped phase')
        fig.colorbar(ax2.imshow(img_unwrapped), ax=ax2)
        ax2.set_title('Unwrapped phase (kamui-graph)')
        if path is not None:
            path = os.path.join(path, 'phase_unwrapped_kamui_graph.png')
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
    return img_unwrapped

'''
