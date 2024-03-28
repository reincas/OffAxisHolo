##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This module provides the function chart(), which delivers a 1951 USAF
# resolution test chart.
#
##########################################################################

import numpy as np
import cairo
from plotfont import Font

# Single element step factor
F = 2**(-1.0/6.0)

# Size of unit field of elements (groups 0 and 1)
SIZE = 3.5 * (F + F**2 + F**3 + F**4) + 2.5 * F**5

# Enumeration of element anchor corners
A_UL, A_LL, A_LR, A_UR = range(4)


def frame(ctx):
    
    """ Draw a line frame around the current field of elements for debugging
    purposes."""
    
    lw = 0.1
    ctx.set_line_width(lw)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.set_line_join(cairo.LineJoin.ROUND)
    #x0 = -1.8/(2**group * size)
    #y0 = -2.5/(2**group * size)
    x0 = -1.8
    y0 = -2.5
    x1 = SIZE - x0
    y1 = SIZE - y0
    ctx.move_to(x0, y0)
    ctx.line_to(x1, y0)
    ctx.line_to(x1, y1)
    ctx.line_to(x0, y1)
    ctx.line_to(x0, y0)
    ctx.stroke()


def text(ctx, text, lw, args):
    
    """ Write text with given line width and font arguments at the current
    position. """
    
    # Save context status
    ctx.save()
    
    # Get text strokes
    font = Font(**args)
    lines = font.string(text)
    #bbox = font.bbox(lines)
    
    # Draw text strokes
    ctx.set_line_width(lw)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.set_line_join(cairo.LineJoin.ROUND)
    for line in lines:
        ctx.move_to(*line[0])
        for x, y in line[1:]:
            ctx.line_to(x, y)
        ctx.stroke()

    # Restore context status
    ctx.restore()


def title(ctx):
    
    """ Write title text 'USAF-1951' centered below the current field of
    elements. """
    
    # Save context status
    ctx.save()
    
    # Font parameters
    lw = 0.2*F**-1
    fs = 0.8 * (2.5*F**3 - lw)
    args = {
        "size": fs,
        "width": fs,
        "halign": "center",
        "valign": "bottom",
        "mirrory": True,
        "mirrorx": False,
        }
    
    # Title text
    ctx.translate(0.5*SIZE, SIZE + 1.0)
    text(ctx, "USAF-1951", lw, args)

    # Restore context status
    ctx.restore()


def element(ctx, element, anchor=A_UL, mirror=False):
    
    """ Draw test target field with unit period consisting of three
    horizontal and three vertical bars with a period of 1.0. Mark the element
    with the given number besides the horizontal bars. """
    
    assert anchor in (A_UL, A_LL, A_LR, A_UR)
    mirror = bool(mirror)
    
    # Save context status
    ctx.save()
    
    # Optional mirror transform
    if mirror:
        ctx.transform(cairo.Matrix(-1, 0, 0, 1, 0, 0))
        anchor = (A_UR, A_LR, A_LL, A_UL)[anchor]
    
    # Anchor point transform
    if anchor == A_LL:
        ctx.translate(0.0, -2.5)
    if anchor == A_LR:
        ctx.translate(-6.0, -2.5)
    if anchor == A_UR:
        ctx.translate(-6.0, 0.0)
    
    # Fill color
    #ctx.set_source_rgb(0.0, 0.0, 0.0)

    # Draw horizontal bars
    for i in range(3):
        ctx.rectangle(0.0, i, 2.5, 0.5)
        ctx.fill()
    
    # Draw vertical bars
    for i in range(3):
        ctx.rectangle(3.5+i, 0.0, 0.5, 2.5)
        ctx.fill()

    # Write given number
    lw = 0.2 / F**(element-3)
    fs = 2.5 / F**(element-7) - lw
    args = {
        "size": fs,
        "width": fs,
        "halign": "right",
        "valign": "center",
        "mirrory": True,
        "mirrorx": mirror,
        }
    ctx.translate(-1.0, 1.25)
    text(ctx, str(element), lw, args)
    
    # Restore context status
    ctx.restore()
    

def field(ctx, group):
    
    """ Draw a quadratic field with all six elements of the given and all six
    of the following group. Mark both groups above the field and all elements
    besides the field. """
    
    # Left column: elements 2-6 of group
    ctx.save()
    for i in range(2, 7):
        if i > 2:
            ctx.translate(0, 3.5)
        ctx.scale(F, F)
        element(ctx, i, A_UL, False)
    ctx.restore()
    
    # Right column: elements 1-6 of group+1
    ctx.save()
    ctx.translate(SIZE, 0.0)
    ctx.scale(0.5, 0.5)
    for i in range(1, 7):
        if i > 1:
            ctx.translate(0, 3.5)
            ctx.scale(F, F)
        element(ctx, i, A_UR, True)
    ctx.restore()
    
    # Bottom right element: element 1 of group
    ctx.save()
    ctx.translate(SIZE, SIZE)
    element(ctx, 1, A_LR, True)
    ctx.restore()

    # Quadratic field
    w = SIZE - 7.0*F - 3.5
    ctx.rectangle(7.0*F, 0.0, w, w)
    ctx.fill()

    # Font parameters
    lw = 0.2*F**-1
    fs = 0.8 * (2.5*F**3 - lw)
    args = {
        "size": fs,
        "width": fs,
        "halign": "center",
        "valign": "top",
        "mirrory": True,
        "mirrorx": False,
        }

    # Left number: group
    ctx.save()
    ctx.translate(3.0, -1.0)
    text(ctx, str(group), lw, args)
    ctx.restore()

    # Right number: group+1
    ctx.save()
    ctx.translate(SIZE-1.5, -1.0)
    text(ctx, str(group+1), lw, args)
    ctx.restore()


def fields(ctx, pitch, w, h):
    
    """ Draw largest field for which all elements fit ito the given image size
    and recursively draw all sub-fields until the period of the smallest
    element is less than one pixel. The pitch is given in mm/px. """
    
    # Save context status
    ctx.save()
 
    # Largest group for which all elements fit ito the given image size
    group = int(np.ceil(np.log(SIZE / (pitch * min(h,w))) / np.log(2.0)))
    
    # Scale of this group
    scale = 2**group
    
    # Period of the smallest element of group+1
    res = 2**(-11/6) / (pitch*scale)
    
    # Place the field of elements in the center of the image
    dx = 0.5 * (w - SIZE / (scale*pitch))
    dy = 0.5 * (h - SIZE / (scale*pitch))
    ctx.translate(dx, dy)
    
    # Scale according to the group
    ctx.scale(1.0/(scale*pitch), 1.0/(scale*pitch))
    #frame(ctx)
    
    # Write title of the test chart
    title(ctx)
    
    # Draw the field of elements for group and group+1
    #print("Group: %d" % group)
    #print("Resolution: %.1f px" % res)
    field(ctx, group)
    
    # Draw the sub-fields
    while res > 1.0:
    
        # Place the sub-field
        dx = 6.5*F
        dy = (3.5*F + 0.5*2.5*F**2)
        ctx.translate(dx, dy)
        
        # Increment group number and scale context
        group += 2
        scale = 2**group
        res = 2**(-11/6) / (pitch*scale)
        ctx.scale(0.25, 0.25)
        
        # Draw the field of elements for the sub-field
        #print("Group: %d" % group)
        #print("Resolution: %.1f px" % res)
        field(ctx, group)

    # Restore context status
    ctx.restore()


def chart(pitch, w, h):

    """ Return an 8-bit grayscale image containing a 1951 USAF resolution test
    chart with the largest field for which all elements fit ito the given image
    size and all sub-fields until the period of the smallest element is less
    than one pixel. The pixel pitch is given in µm/px. """
    
    # Image surface
    surface = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    
    # Drawing context
    ctx = cairo.Context(surface)
    
    # Draw the chart
    fields(ctx, 1e-3*pitch, w, h)
    
    # Convert the drawing into a 8-bit grayscale image    
    data = surface.get_data()
    img = np.ndarray(shape=(h, w), dtype=np.uint8, buffer=data)
    
    # Return the image
    return img
