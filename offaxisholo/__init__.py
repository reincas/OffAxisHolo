##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This package provides tools for the reconstruction and simulation of
# off-axis holograms.
#
##########################################################################

from pathlib import Path
from shutil import rmtree

from .usaf1951 import chart as chart_usaf
from . import image
from . import objects
from . import field
from . import simulate
from . import reconstruct
from .zernike import Zernike


def mkdir(path, clean=True):

    """ Make sure that the given folder exists and is empty. """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if clean:
        for sub in p.iterdir():
            if sub.is_file():
                sub.unlink()
            elif sub.is_dir():
                rmtree(sub)
    return path
