##########################################################################
# Copyright (c) 2022-2024 Reinhard Caspary                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from offaxisholo import mkdir, image, objects, field, simulate, reconstruct


##########################################################################
# Parameters
##########################################################################

# Size of quadratic hologram in pixel
N = 1024

# Wavelength in µm
wl = 0.675

# Object pixel pitch in wavelength units
po = 0.276 / wl

# Aperture radius in wavelength units
ra = 2970 / wl

# Focal length of microscope objective in wavelength units
fmo = 8250 / wl

# Focal length of tube lens in wavelength units
ftl = 175000 / wl

# Distance of tube lens from back focal plane of objective
# in wavelength units
dtl = 1.0 * ftl

# Distance of camera sensor from back focal plane of tube lens
# in wavelength units
dc = 2000.0 / wl

# Polar and azimutal angles of plane reference wave 
opt = True
if opt:
    fx = N // 4 
    fy = N // 4 
    theta = np.arcsin(np.sqrt(fx*fx + fy*fy) / N)
    phi = np.arctan2(fy, fx)
else:
    theta = 28.006 * np.pi/180.0
    phi = 45.0 * np.pi/180.0


##########################################################################
# Simulation
##########################################################################

# Illumination field
Fp = field.planar(N, 0, 0)
#Fp = field.spherical(N, po, 20000.0 / wl)

# Object field
Fo = Fp * objects.asphase(objects.usaf(N, po*wl), 0.2)

# Aperture plane field (back focal plane of objective)
Fs, ps = simulate.lens(Fo, po, fmo)
Fs = field.norm(Fs)
Fs *= field.aperture(N, ra, ps)

# Image field (back focal plane of tube lens)
Fi, pi = simulate.lens(Fs, ps, ftl, dtl)
Fi = field.norm(Fi)

# Camera field with reference wave
Fc, pc = simulate.propagate(Fi, pi, dc)
Fc = field.norm(Fc)
Fc += 1.0 * field.planar(N, theta, phi)
Fc = field.norm(Fc)

# Hologram image
holo = np.abs(Fc)


##########################################################################
# Reconstruction
##########################################################################

# Spectrum of camera image (magnitude of field)
Sc, fx, fy, weight = reconstruct.locateOrder(holo, 16)

# Filtered camera spectrum
if opt:
    r = np.sqrt(fx*fx + fy*fy) - 5
else:
    r = 1.0 * ra/ps
Sr = reconstruct.rollImage(Sc, fx, fy)
Sr = reconstruct.circularMask(Sr, r)

# Reconstructed field with back propagation to image plane
Fr = reconstruct.getField(Sr)
Fr, pr = simulate.propagate(Fr, pc, -dc)


##########################################################################
# Results
##########################################################################

print("Object pitch:     %.3f um" % (wl*po))
print("Spectrum pitch:   %.3f um" % (wl*ps))
print("Apertur diameter: %.1f um (%d px)" % (2*ra*wl, 2*ra/ps))
print("Camera pitch:     %.3f um" % (wl*pi))
print("Order offset:     %d, %d px" % (fx, fy))
print("DC diameter:      %d px" % (4*ra/ps))

# List of fields and spectra to be displayed
fields = [Fo, Fs, Fi, Fc, Sc, Sr, Fr]

# Prepare magnitude arrays
mag = [np.abs(F) for F in fields]
mag[1] = np.log(mag[1] + 1e-7*np.max(mag[1]))
mag[4] = np.log(mag[4] + 1e-7*np.max(mag[4]))
mag[5] = np.log(mag[5] + 1e-7*np.max(mag[5]))
mag = [image.normcolor(F) for F in mag]
mag[1] = image.drawCircle(mag[1], 0, 0, round(ra/ps), image.CV_RED, 2)
mag[4] = image.drawCircle(mag[4], 0, 0, round(2*ra/ps), image.CV_RED, 2)
mag[4] = image.drawCircle(mag[4], -fx, -fy, round(ra/ps), image.CV_RED, 2)
mag[4] = image.drawCircle(mag[4], fx, fy, round(ra/ps), image.CV_RED, 2)
mag[4] = image.drawCircle(mag[4], -fx, -fy, round(r), image.CV_GREEN, 2)
mag[4] = image.drawCircle(mag[4], fx, fy, round(r), image.CV_GREEN, 2)
mag = np.concatenate(mag, axis=1)

# Prepare phase arrays
ang = [np.angle(F) for F in fields]
#ang[6] = unwrap_phase(ang[6])
#ang[6] = image.blur(ang[6], 1)
ang = [F/(2*np.pi) for F in ang]
ang = [(F-F.mean()) + 0.5 for F in ang]
#ang = [(F-F[0,0]) + 0.5 for F in ang]
ang = [image.normcolor(F, False) for F in ang]
ang[1] = image.drawCircle(ang[1], 0, 0, round(ra/ps), image.CV_RED, 2)
ang = np.concatenate(ang, axis=1)

# Concatenate all arrays
img = np.concatenate((mag, ang), axis=0)

# Store result in file
path = mkdir("test/dhm")
file = Path(path, "dhm.png")
image.write(file, img)

# Display result
plt.close()
if 0:
    X = np.arange(N)
    line = 246
    Y1 = np.angle(Fo) / (2*np.pi)
    Y2 = np.angle(Fr) / (2*np.pi)
    Y2 = image.blur(Y2, 1)
    Y1 = Y1[-line-1,:]
    Y2 = Y2[line,::-1]
    Y1 -= Y1.mean()
    Y2 -= Y2.mean()
    plt.plot(X, Y1, X, Y2, X, 0*X)
else:
    plt.imshow(img[:,:,::-1])
plt.show()

