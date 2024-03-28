##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import glob
import shutil

GLOBS = ["./build/", "./dist/", "./*.egg-info/", "./**/__pycache__/"]

def clean(globs=None):

    if globs is None:
        globs = GLOBS
        
    for pattern in globs:
        for path in glob.iglob(pattern):
            print("Removing '%s'" % path)
            shutil.rmtree(path)


if __name__ == '__main__':

    clean()
