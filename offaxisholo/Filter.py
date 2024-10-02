import numpy as np


class Filter:
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply a filter to the image."""
        raise NotImplementedError("Filter method not implemented.")


class GaussianFilter(Filter):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filter to the image."""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(image, sigma=self.sigma)


class HM2F(Filter):
    def __init__(self, sigma: float):
        self.sigma = sigma
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
