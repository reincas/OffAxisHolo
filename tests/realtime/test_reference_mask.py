"""A-priori reference mask from occupied_um polygons (K19, G10 SA2)."""

import math

import numpy as np
import pytest

from offaxisholo.realtime import reference_mask
from offaxisholo.realtime.reference_mask import pixel_positions_um

PIXEL = (0.25, 0.25)
SHAPE = (256, 256)  # 64 µm field of view
CENTER = (850.0, 1200.0)  # stage (y, x) of the image centre


def square(center, side):
    cy, cx = center
    h = side / 2
    return [[cy - h, cx - h], [cy - h, cx + h], [cy + h, cx + h], [cy + h, cx - h]]


def occupied_area(mask, pixel=PIXEL):
    return np.count_nonzero(~mask) * pixel[0] * pixel[1]


def mask(polygons, margin=0.0, **kwargs):
    return reference_mask(
        SHAPE,
        PIXEL,
        CENTER,
        occupied_um=polygons,
        pose_tolerance_um=margin,
        optical_margin_um=0.0,
        **kwargs,
    )


def test_pixel_centres_are_symmetric_around_the_stage_position():
    y, x = pixel_positions_um((4, 6), (0.5, 0.25), (10.0, 20.0))
    assert y.mean() == pytest.approx(10.0) and x.mean() == pytest.approx(20.0)
    assert y[1, 0] - y[0, 0] == pytest.approx(0.5)
    assert x[0, 1] - x[0, 0] == pytest.approx(0.25)


def test_nothing_printed_gives_a_full_reference():
    result = mask([])
    assert result.dtype == bool and result.shape == SHAPE and result.all()


def test_exact_area_within_one_pixel_layer():
    side = 20.0
    result = mask([square(CENTER, side)])
    perimeter = 4 * side
    assert occupied_area(result) == pytest.approx(side**2, abs=perimeter * PIXEL[0])
    # the centre is occupied, the corners of the field are free
    assert not result[128, 128] and result[0, 0] and result[-1, -1]


def test_position_follows_the_stage():
    """A square right of the image centre appears at larger column indices."""
    result = mask([square((CENTER[0], CENTER[1] + 20.0), 8.0)])
    cols = np.nonzero(~result)[1]
    assert cols.min() > 128 + (20.0 - 4.0) / PIXEL[1] - 2
    assert cols.max() < 128 + (20.0 + 4.0) / PIXEL[1] + 2


def test_margin_dilation_matches_the_minkowski_area():
    side, margin = 10.0, 2.0
    pose, optical = 1.5, 0.5
    result = reference_mask(
        SHAPE,
        PIXEL,
        CENTER,
        occupied_um=[square(CENTER, side)],
        pose_tolerance_um=pose,
        optical_margin_um=optical,
    )
    expected = side**2 + 4 * side * margin + math.pi * margin**2
    perimeter = 4 * side + 2 * math.pi * margin
    assert occupied_area(result) == pytest.approx(expected, abs=perimeter * PIXEL[0])


def test_rotation_by_the_calibration():
    """A rectangle rotated with the image -> stage matrix keeps its area; a point
    on its long axis moves accordingly."""
    long, short = 30.0, 6.0
    rect = [
        [CENTER[0] - short / 2, CENTER[1] - long / 2],
        [CENTER[0] - short / 2, CENTER[1] + long / 2],
        [CENTER[0] + short / 2, CENTER[1] + long / 2],
        [CENTER[0] + short / 2, CENTER[1] - long / 2],
    ]
    angle = math.radians(90)
    rotation = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    plain = mask([rect])
    rotated = mask([rect], image_to_stage=rotation)
    assert occupied_area(rotated) == pytest.approx(occupied_area(plain), rel=0.05)
    # 12 µm along x in the stage is 12 µm along y in the rotated image
    step = int(12.0 / PIXEL[0])
    assert not plain[128, 128 + step] and plain[128 + step, 128]
    assert not rotated[128 + step, 128] and not rotated[128 - step, 128]
    assert rotated[128, 128 + step]


def test_several_polygons_and_a_concave_one():
    l_shape = [
        [CENTER[0] - 10, CENTER[1] - 10],
        [CENTER[0] - 10, CENTER[1] + 10],
        [CENTER[0] - 5, CENTER[1] + 10],
        [CENTER[0] - 5, CENTER[1] - 5],
        [CENTER[0] + 10, CENTER[1] - 5],
        [CENTER[0] + 10, CENTER[1] - 10],
    ]
    result = mask([l_shape, square((CENTER[0] + 20, CENTER[1] + 20), 4.0)])
    expected = 20 * 5 + 15 * 5 + 16
    assert occupied_area(result) == pytest.approx(expected, abs=0.25 * 100)
    assert result[128 + 20, 128 + 20]  # inside the notch of the L: free


def test_raster_alternative_matches_the_polygon():
    side = 20.0
    grid = {"origin_um": [CENTER[0] - 40.0, CENTER[1] - 40.0], "pixel_um": [0.25, 0.25]}
    raster = np.zeros((321, 321), dtype=np.uint8)
    y, x = np.meshgrid(
        grid["origin_um"][0] + 0.25 * np.arange(321),
        grid["origin_um"][1] + 0.25 * np.arange(321),
        indexing="ij",
    )
    raster[
        (np.abs(y - CENTER[0]) <= side / 2) & (np.abs(x - CENTER[1]) <= side / 2)
    ] = 1
    for margin in (0.0, 2.0):
        from_raster = reference_mask(
            SHAPE,
            PIXEL,
            CENTER,
            occupied_raster=raster,
            raster_grid=grid,
            pose_tolerance_um=margin,
            optical_margin_um=0.0,
        )
        from_polygon = mask([square(CENTER, side)], margin=margin)
        differ = np.count_nonzero(from_raster != from_polygon)
        assert differ <= 4 * (side + 2 * margin) / PIXEL[0] * 2  # boundary pixels only


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        ({"occupied_um": [], "occupied_raster": np.zeros((2, 2))}, "exactly one"),
        ({"occupied_raster": np.zeros((2, 2))}, "raster_grid"),
        ({"occupied_um": [[[0, 0], [1, 1]]]}, "3 vertices"),
        ({"occupied_um": [], "pose_tolerance_um": -1.0}, ">= 0"),
    ],
)
def test_invalid_input(kwargs, message):
    params = {"pose_tolerance_um": 0.0, "optical_margin_um": 0.0, **kwargs}
    with pytest.raises(ValueError, match=message):
        reference_mask(SHAPE, PIXEL, CENTER, **params)
