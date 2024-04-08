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

# Focal length of microscope objective in wavelength units
fmo = 8250 / wl

# Pupil radius of microscope objective in wavelength units
ra = 2970 / wl  # External pupil!?
#ra = 6500 / wl  # Pupil of microscope objective

# Focal length of tube lens in wavelength units
ftl = 175000 / wl

# Distance of tube lens to back focal plane of objective in wavelength units
dtl = 1.0 * ftl

# Distance of sensor to back focal plane of tube lens in wavelength units
ds = 2000.0 / wl

# Off-axis DHM configuration
params = {
    "illuminationField": field.planar(N, 0, 0),
    "objectiveFocalLength": fmo,
    "pupilRadius": ra,
    "tubeFocalLength": ftl,
    "tubeDistance": dtl,
    "sensorDistance": ds,
    }
opt = True
if opt:
    params["referencePosition"] = N // 4
else:
    theta = 28.006 * np.pi/180.0
    phi = 45.0 * np.pi/180.0
    params["referenceTilt"] = (theta, phi)
    
# Off-axis DHM object
dhm = simulate.HoloMicroscope(**params)


##########################################################################
# Simulation
##########################################################################

# Object transmission array and pixel pitch in wavelength units
po = 0.276 / wl
obj = objects.asphase(objects.usaf(N, po*wl), 0.2)
#obj = np.zeros((N, N), dtype=complex)
#obj[N//2-3, N//2-3] = 1.0
#obj[N//2+2, N//2+2] = 1.0

# Recorded hologram, hologram pitch and dictionary of all fields
holo, ph, fields = dhm.hologram(obj, po, fields=True)


##########################################################################
# Reconstruction
##########################################################################

# Spectrum of camera image (magnitude of field)
Sc, fx, fy, weight = reconstruct.locateOrder(holo, 16)

# Filtered camera spectrum
if opt:
    r = np.sqrt(fx*fx + fy*fy) - 5
else:
    pa = fmo / (N*po)
    r = 1.0 * ra/pa
Sr = reconstruct.rollImage(Sc, fx, fy)
Sr = reconstruct.circularMask(Sr, r)

# Reconstructed field with back propagation to image plane
pc = po * ftl / fmo
Fr = reconstruct.getField(Sr)
Fr = simulate.propagate(Fr, pc, -ds)


##########################################################################
# Results
##########################################################################

Fo, po = fields["object"]
Fa, pa = fields["pupil"]
Fi, pi = fields["image"]
Fc, pc = fields["sensor"]

print("Object pitch:     %.3f um" % (wl*po))
print("Spectrum pitch:   %.3f um" % (wl*pa))
print("Apertur diameter: %.1f um (%d px)" % (2*ra*wl, 2*ra/pa))
print("Camera pitch:     %.3f um" % (wl*pi))
print("Order offset:     %d, %d px" % (fx, fy))
print("DC diameter:      %d px" % (4*ra/pa))

# List of fields and spectra to be displayed
fields = [Fo, Fa, Fi, Fc, Sc, Sr, Fr]

# Prepare magnitude arrays
mag = [np.abs(F) for F in fields]
mag[1] = np.log(mag[1] + 1e-7*np.max(mag[1]))
mag[4] = np.log(mag[4] + 1e-7*np.max(mag[4]))
mag[5] = np.log(mag[5] + 1e-7*np.max(mag[5]))
mag = [image.normcolor(F) for F in mag]
mag[1] = image.drawCircle(mag[1], 0, 0, round(ra/pa), image.CV_RED, 2)
mag[4] = image.drawCircle(mag[4], 0, 0, round(2*ra/pa), image.CV_RED, 2)
mag[4] = image.drawCircle(mag[4], -fx, -fy, round(ra/pa), image.CV_RED, 2)
mag[4] = image.drawCircle(mag[4], fx, fy, round(ra/pa), image.CV_RED, 2)
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
ang[1] = image.drawCircle(ang[1], 0, 0, round(ra/pa), image.CV_RED, 2)
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

