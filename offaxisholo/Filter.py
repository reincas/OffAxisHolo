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

