"""
A-priori reference mask from the printed geometry (K19, decision G10 SA2).

nanofactorysystem sends ``occupied_um`` with every capture: polygons in stage
µm of all printed matter in the field of view (or a ``uint8`` raster). The
DHMConnect service turns them into a boolean mask on the reconstruction grid
with :func:`reference_mask` (True = structure-free = usable as reference for
piston, tilt and noise). The margin around the structures is
``pose_tolerance_um`` (capture context) + ``optical_margin_um`` (parameter
set).

Geometry: pixel ``(i, j)`` of an ``(ny, nx)`` grid with pixel size
``(dy, dx)`` lies at the image offset ``((i - (ny - 1) / 2) * dy,
(j - (nx - 1) / 2) * dx)`` µm from the image centre. ``image_to_stage`` (2 x 2,
``[y, x]`` order) maps this offset to a stage offset (rotation, scale, shear
from the calibration T104 / MI-41; default identity, i.e. rotation 0 and the
nominal pixel size until the calibration exists). The image centre is at the
capture's actual stage position.

Pure NumPy, host only (it runs once per capture, not per frame); no Qt, no
CuPy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["pixel_positions_um", "reference_mask"]


def pixel_positions_um(shape, pixel_um, center_um, image_to_stage=None):
    """Stage positions of the pixel centres of a grid.

    Parameters
    ----------
    shape : tuple of int
        ``(ny, nx)``.
    pixel_um : tuple of float
        Pixel size ``(dy, dx)`` in µm.
    center_um : tuple of float
        Stage position ``(y, x)`` of the image centre in µm.
    image_to_stage : array_like, optional
        2 x 2 matrix (``[y, x]`` order) from image offsets to stage offsets;
        default identity.

    Returns
    -------
    y_um, x_um : numpy.ndarray
        Stage coordinates, each of shape ``shape``.
    """
    ny, nx = (int(n) for n in shape)
    dy, dx = (float(p) for p in pixel_um)
    if ny < 1 or nx < 1 or dy <= 0 or dx <= 0:
        raise ValueError(f"Invalid grid: shape {shape}, pixel_um {pixel_um}.")
    matrix = np.eye(2) if image_to_stage is None else np.asarray(image_to_stage, float)
    if matrix.shape != (2, 2):
        raise ValueError(f"image_to_stage must be 2 x 2, got {matrix.shape}.")
    off_y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    off_x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    oy, ox = np.meshgrid(off_y, off_x, indexing="ij")
    y_um = center_um[0] + matrix[0, 0] * oy + matrix[0, 1] * ox
    x_um = center_um[1] + matrix[1, 0] * oy + matrix[1, 1] * ox
    return y_um, x_um


def _polygon_inside_and_distance(y, x, polygon):
    """Even-odd inside test and distance to the boundary of one polygon."""
    vertices = np.asarray(polygon, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("A polygon needs at least 3 vertices [y, x].")
    inside = np.zeros(y.shape, dtype=bool)
    distance_sq = np.full(y.shape, np.inf)
    for (y0, x0), (y1, x1) in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        # even-odd rule: a horizontal ray towards +x crosses this edge
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < x_cross)
        # distance to the segment
        ey, ex = y1 - y0, x1 - x0
        length_sq = ey * ey + ex * ex
        if length_sq == 0:
            t = np.zeros(y.shape)
        else:
            t = np.clip(((y - y0) * ey + (x - x0) * ex) / length_sq, 0.0, 1.0)
        dist = (y - (y0 + t * ey)) ** 2 + (x - (x0 + t * ex)) ** 2
        np.minimum(distance_sq, dist, out=distance_sq)
    return inside, np.sqrt(distance_sq)


def _raster_occupied(y, x, raster, raster_grid, margin_um):
    """Occupied pixels of the target grid from a ``uint8`` raster plus margin."""
    from scipy import ndimage  # only for the raster alternative

    raster = np.asarray(raster)
    if raster.ndim != 2:
        raise ValueError("The occupied raster must be 2-D.")
    origin = [float(v) for v in raster_grid["origin_um"]]
    pitch = [float(v) for v in raster_grid["pixel_um"]]
    occupied = raster != 0
    if margin_um > 0:
        # distance (µm) of every raster pixel to the nearest occupied one
        distance = ndimage.distance_transform_edt(~occupied, sampling=pitch)
        occupied = distance <= margin_um
    iy = np.rint((y - origin[0]) / pitch[0]).astype(np.int64)
    ix = np.rint((x - origin[1]) / pitch[1]).astype(np.int64)
    valid = (iy >= 0) & (iy < raster.shape[0]) & (ix >= 0) & (ix < raster.shape[1])
    result = np.zeros(y.shape, dtype=bool)
    result[valid] = occupied[iy[valid], ix[valid]]
    return result


def reference_mask(
    shape,
    pixel_um,
    center_um,
    *,
    occupied_um: Sequence | None = None,
    occupied_raster=None,
    raster_grid: dict | None = None,
    pose_tolerance_um: float,
    optical_margin_um: float,
    image_to_stage=None,
) -> np.ndarray:
    """Boolean reference mask on the reconstruction grid (True = structure-free).

    Parameters
    ----------
    shape : tuple of int
        Grid ``(ny, nx)`` of the reconstruction (after the sideband crop).
    pixel_um : tuple of float
        Pixel size ``(dy, dx)`` of that grid in µm (``ReconstructionResult.pixel_um``).
    center_um : tuple of float
        Actual stage position ``(y, x)`` of the image centre in µm.
    occupied_um : sequence of polygons, optional
        Polygons in stage µm, each a list of ``[y, x]`` vertices (contract
        ``CaptureContext.occupied_um``); ``[]`` = nothing printed.
    occupied_raster : array_like, optional
        Alternative: ``uint8`` raster (non-zero = occupied) on ``raster_grid``
        (``{"origin_um": [y, x], "pixel_um": [dy, dx]}``, origin = centre of
        pixel (0, 0)). Pixels outside the raster count as free.
    raster_grid : dict, optional
        Grid of ``occupied_raster``.
    pose_tolerance_um : float
        Stage pose tolerance (capture context), added to the margin.
    optical_margin_um : float
        Optical margin (parameter set), added to the margin.
    image_to_stage : array_like, optional
        2 x 2 image -> stage matrix (calibration); default identity.

    Returns
    -------
    numpy.ndarray of bool
        Shape ``shape``; True where no printed matter lies within the margin.

    Raises
    ------
    ValueError
        If neither or both geometry inputs are given, or for a negative
        margin, an invalid grid or an invalid polygon.
    """
    if (occupied_um is None) == (occupied_raster is None):
        raise ValueError("Give exactly one of occupied_um and occupied_raster.")
    margin = float(pose_tolerance_um) + float(optical_margin_um)
    if pose_tolerance_um < 0 or optical_margin_um < 0:
        raise ValueError("pose_tolerance_um and optical_margin_um must be >= 0.")
    y, x = pixel_positions_um(shape, pixel_um, center_um, image_to_stage)
    if occupied_raster is not None:
        if raster_grid is None:
            raise ValueError("occupied_raster needs raster_grid.")
        return ~_raster_occupied(y, x, occupied_raster, raster_grid, margin)
    occupied = np.zeros(y.shape, dtype=bool)
    for polygon in occupied_um:
        inside, distance = _polygon_inside_and_distance(y, x, polygon)
        occupied |= inside | (distance <= margin)
    return ~occupied
