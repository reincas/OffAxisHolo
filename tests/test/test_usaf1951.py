##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from pathlib import Path
import matplotlib.pyplot as plt
from offaxisholo import image, mkdir, chart_usaf

h, w = 1024, 1024
pitch = 0.276
img = chart_usaf(pitch, w, h)

path = mkdir("test/usaf1951")
file = Path(path, "usaf1951.png")
image.write(file, img)

plt.close()
plt.imshow(img)
plt.show()
