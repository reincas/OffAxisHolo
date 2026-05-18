import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase

# from kamui import unwrap_dimensional


def phase_unwrapping_fast2d(phase_wrapped):
    """
    This phase unwrapping algorithm is based on:
    Miguel Arevallilo Herráez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat, "Fast two-dimensional
    phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path,"
    Appl. Opt. 41, 7437-7444 (2002)
    https://doi.org/10.1364/AO.41.007437
    """
    phase_unwrapped = unwrap_phase(phase_wrapped)
    return phase_unwrapped


def phase_unwrapping_wrap_around(phase_wrapped, plot=True, path=None):
    """
    Same as fast 2d phase unwrapping with non-continuous path, but with wrap around.
    https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html#id1

    """
    # Unwrap with wrap around enabled for the 0th dimension
    image_unwrapped_wrap_around = unwrap_phase(phase_wrapped, wrap_around=(True, True))

    if plot:  # todo rework visual
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax1, ax2 = ax.ravel()
        fig.colorbar(ax2.imshow(image_unwrapped_wrap_around), ax=ax2)
        ax2.set_title("Unwrapped phase with wrap around")
        if path is not None:
            path = os.path.join(path, "phase_unwrapped_wrap_around.png")
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
    return image_unwrapped_wrap_around


# NOTE: Kamui is not compatible with numpy 2.xx -> not necessary as i always try to use the fast 2d unwrapping algorithm
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


def phase_unwrapping_numpy(phase_wrapped, plot=True, path=None):
    """
    Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E.
    (2020). Array programming with NumPy. Nature, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
    """
    img_unwrapped = np.unwrap(phase_wrapped)
    if plot:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        # ax1 = ax.ravel()
        fig.colorbar(ax.imshow(img_unwrapped), ax=ax)
        title = "Unwrapped phase with numpy-unwrap"
        ax.set_title(title)
        if path is not None:
            path = os.path.join(path, "phase_unwrapped_numpy.png")
            plt.savefig(path)
            plt.close()
        else:
            plt.show()
    return img_unwrapped
