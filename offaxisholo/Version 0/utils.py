import os
from scidatacontainer import Container


def get_hologram(path, filename=None) -> "img":
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
    return dc._items['meas/image.png']
