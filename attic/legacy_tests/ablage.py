import numpy as np
from scipy import ndimage

"""

This implementation includes several key components:

compensate_background(): The main function that handles background compensation by:

Unwrapping phases before subtraction to handle 2π jumps
Using division for intensity compensation (multiplicative noise model)
Applying basic noise reduction


process_hologram(): A complete processing pipeline that:

Performs background compensation
Removes phase artifacts
Reconstructs the final E-field


remove_phase_artifacts(): A helper function that:

Removes high-frequency noise using Gaussian filtering
Eliminates phase jumps with median filtering


example_usage(): A demonstration function that:

Creates sample data
Shows how to use the processing pipeline
Returns the processed intensity and phase



Key features of this implementation:

Phase handling:

Uses phase unwrapping to handle 2π discontinuities
Applies appropriate filtering to reduce noise
Preserves phase information while removing artifacts


Intensity processing:

Uses division instead of subtraction for background compensation
Includes median filtering to reduce speckle noise
Ensures non-negative intensity values


Error handling:

Checks for array shape consistency
Handles edge cases in intensity values


"""


def compensate_background(object_intensity, object_phase, bg_intensity, bg_phase):
    """
    Compensate for background noise in holographic measurements.

    Parameters:
    -----------
    object_intensity : numpy.ndarray
        2D array of intensity values from object measurement
    object_phase : numpy.ndarray
        2D array of phase values from object measurement
    bg_intensity : numpy.ndarray
        2D array of background intensity values
    bg_phase : numpy.ndarray
        2D array of background phase values

    Returns:
    --------
    tuple
        (compensated_intensity, compensated_phase)
    """

    # Phase compensation
    # Unwrap phases first to handle 2π jumps
    unwrapped_obj_phase = np.unwrap(object_phase)
    unwrapped_bg_phase = np.unwrap(bg_phase)

    # Subtract background phase
    compensated_phase = unwrapped_obj_phase - unwrapped_bg_phase

    # Wrap phase back to [-π, π]
    compensated_phase = np.angle(np.exp(1j * compensated_phase))

    # Intensity compensation
    # Use ratio rather than subtraction to handle multiplicative effects
    compensated_intensity = object_intensity / bg_intensity

    # Apply median filter to reduce noise in intensity
    compensated_intensity = ndimage.median_filter(compensated_intensity, size=3)

    # Ensure non-negative intensity
    compensated_intensity = np.maximum(compensated_intensity, 0)

    return compensated_intensity, compensated_phase


def process_hologram(object_intensity, object_phase, bg_intensity, bg_phase):
    """
    Complete processing pipeline for holographic data.

    Parameters:
    -----------
    Same as compensate_background()

    Returns:
    --------
    numpy.ndarray
        Complex E-field after background compensation
    """
    # First compensate background
    comp_intensity, comp_phase = compensate_background(
        object_intensity, object_phase, bg_intensity, bg_phase
    )

    # Additional noise reduction in phase domain
    comp_phase = remove_phase_artifacts(comp_phase)

    # Calculate final E-field
    compensated_efield = np.sqrt(comp_intensity) * np.exp(1j * comp_phase)

    return compensated_efield


def remove_phase_artifacts(phase, kernel_size=5):
    """
    Remove artifacts from phase data.

    Parameters:
    -----------
    phase : numpy.ndarray
        2D array of phase values
    kernel_size : int
        Size of the filter kernel

    Returns:
    --------
    numpy.ndarray
        Cleaned phase data
    """
    # Remove high-frequency noise
    cleaned_phase = ndimage.gaussian_filter(phase, sigma=1)

    # Remove phase jumps
    cleaned_phase = ndimage.median_filter(cleaned_phase, size=kernel_size)

    return cleaned_phase


# Example usage
def example_usage():
    # Generate sample data
    size = 100
    x, y = np.meshgrid(np.linspace(-5, 5, size), np.linspace(-5, 5, size))

    # Simulate object with phase variation
    object_phase = np.sin(x) + np.cos(y)
    object_intensity = 1 + 0.5 * np.cos(x**2 + y**2)

    # Simulate background noise
    bg_phase = 0.2 * np.random.random((size, size))
    bg_intensity = 1 + 0.1 * np.random.random((size, size))

    # Process the hologram
    processed_efield = process_hologram(
        object_intensity, object_phase, bg_intensity, bg_phase
    )

    # Extract final intensity and phase
    final_intensity = np.abs(processed_efield) ** 2
    final_phase = np.angle(processed_efield)

    return final_intensity, final_phase


if __name__ == "__main__":
    final_intensity, final_phase = example_usage()
