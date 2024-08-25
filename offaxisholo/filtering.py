import numpy as np
from scipy.ndimage import median_filter


def hybrid_median_mean_filter(image, kernel_size=5):
    """
    Methode is chosen based on the following publication:
    Optical Engineering, Vol. 60, Issue 12, 123107 (December 2021). https://doi.org/10.1117/1.OE.60.12.123107
    """
    i = 2
    g_new = image
    while (2*i-1) <= kernel_size:
        # Apply median filter
        h = median_filter(image, size=(2*i-1))
        # Apply mean filter
        g_new = (g_new+h)/2
        i += 1
    return g_new


'''
h = median(g,[3x3])
g^ = (g+h)/2

k =^ max. kernel filter size
    k = 5 -> 2 iterations

'''



