import os

import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase

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
    """
    phase_unwrapped = unwrap_phase(phase_wrapped)
    return phase_unwrapped


def phase_unwrapping_wrap_around(phase_wrapped, plot=False, path=None):
    """
    Same algorithm as phase_unwrapping_fast2d, but with periodic boundary
    conditions (wrap around) along both axes.

    Reference:
        Herraez et al., Appl. Opt. 41, 7437-7444 (2002)
        https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html
    """
    image_unwrapped_wrap_around = unwrap_phase(phase_wrapped, wrap_around=(True, True))

    if plot:
        fig, ax = plt.subplots(1, 1)
        fig.colorbar(ax.imshow(image_unwrapped_wrap_around), ax=ax)
        ax.set_title("Unwrapped phase with wrap around")
        if path is not None:
            fig.savefig(os.path.join(path, "phase_unwrapped_wrap_around.png"))
            plt.close(fig)
        else:
            plt.show()
    return image_unwrapped_wrap_around


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
