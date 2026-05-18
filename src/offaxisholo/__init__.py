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
import glob
from pathlib import Path
import os
from scidatacontainer import Container
import logging
from shutil import rmtree
import numpy as np
# from .docker import DockerBase as Docker
# from .reconstruction import Hologram, ReferenceHologram, HologramProcessor
# from .reconstruction import DHMPlotter
from .src.offaxisholo.loader import DataLoader

LOGFMT = logging.Formatter(fmt="%(asctime)s / %(levelname)s / %(message)s",
                           datefmt="%Y-%m-%d %H:%M:%S")


def mkdir(path, clean=False):
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

def get_logger(logfile=None):
    """ Configure and return a logger object. """

    # Initialize logger object
    logger = logging.getLogger('dummy')
    logger.setLevel(logging.DEBUG)

    # Console output
    consolehandler = logging.StreamHandler()
    consolehandler.setLevel(logging.DEBUG)
    consolehandler.setFormatter(LOGFMT)
    logger.addHandler(consolehandler)

    # Optional file output
    if logfile:
        filehandler = logging.FileHandler(logfile)
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(LOGFMT)
        logger.addHandler(filehandler)

    # Return logger object
    return logger


def get_hologram(path, filename=None, img_container=False) -> np.ndarray:
    """
    Returns the SciDataContainer PNGFileContainer of the Hologram.
    Path: Path of the ZDC Container.#
            With the Name of the ZDC_Container in the Path variable
    OR  filename="name of ZDC-Container"
    AND Path = Directory of the file.
    """
    if filename is None:
        if path[-4:] == ".zdc":
            dc = Container(file=path)
        else:
            path = path + ".zdc"
            try:
                dc = Container(file=path)
            except FileNotFoundError:
                raise FileNotFoundError(f"No file at {path} found.")
    else:
        path = os.path.join(path, filename)
        try:
            dc = Container(file=path)
        except Exception as e:
            print(f"{type(e)}: {e}")

    if img_container:
        return dc._items['meas/image.png']
    else:
        return dc._items['meas/image.png'].data


def get_datafiles(root, subdir=False, ending='.dat') -> list:
    if subdir:
        files='**\*'+ending
        path = os.path.join(root, files)
    else:
        files = '*' + ending
        path = os.path.join(root, files)
    try:
        return glob.glob(path)
    except Exception as e:
        raise FileNotFoundError(f"No directory {root}. Exception {e}")
