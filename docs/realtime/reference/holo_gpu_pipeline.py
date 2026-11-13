"""Off-axis-DHM: Demodulation, Hintergrundkompensation, Mittelung, Propagation.

Läuft mit CuPy auf der GPU; ohne CuPy automatisch mit NumPy/SciPy auf der CPU
(gleicher Code, gleiche Ergebnisse -> bequem zum Testen/Debuggen).

Annahmen: quadratisches Kamerabild (N x N), Träger bekannt (cycles/px),
das -1. Beugungsordnung-Sideband enthält das Objektfeld O (Term R* O).
Der absolute Amplitudenmaßstab (M/N)^2 der IFFT wird weggelassen; er kürzt sich in
der Hintergrunddivision.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import tukey

try:                                    # GPU-Pfad
    import cupy as xp
    import cupyx.scipy.fft as xfft
    GPU = True
except ImportError:                     # CPU-Fallback
    import scipy.fft as xfft
    xp = np
    GPU = False


class OffAxisPipeline:
    def __init__(self, n, crop, carrier, pixel_size_um, wavelength_um, n_medium=1.0,
                 ref_mask=None, eps_rel=1e-3, edge_alpha=0.1, sideband_alpha=0.5,
                 mode="block", ema_alpha=0.05, window=32):
        self.n, self.m = n, crop
        self.dx_c = pixel_size_um * n / crop             # Pixelabstand des Crop-Felds
        self.lam = wavelength_um / n_medium
        self.eps_rel, self.mode, self.alpha, self.k = eps_rel, mode, ema_alpha, window

        w = tukey(n, edge_alpha)
        self.win = xp.asarray(np.outer(w, w), dtype=xp.float32)
        # Crop-Indizes direkt in IFFT-Reihenfolge (0, 1, ..., -1) -> kein fftshift nötig
        off = np.fft.ifftshift(np.arange(-crop // 2, crop // 2))
        cx, cy = -round(carrier[0] * n), -round(carrier[1] * n)
        self.ix = xp.ix_(xp.asarray((cy + off) % n), xp.asarray((cx + off) % n))
        sb = tukey(crop, sideband_alpha)
        self.sbwin = xp.asarray(np.fft.ifftshift(np.outer(sb, sb)), dtype=xp.float32)

        # Referenzbereich als Float-Gewicht statt Bool-Indexing (vermeidet Host-Sync)
        m = np.ones((crop, crop), np.float32) if ref_mask is None else np.asarray(ref_mask, np.float32)
        self.ref_w = xp.asarray(m)

        self._H = {}
        self.Ub = self.Ub_inv = None
        self.reset()

    # ---------------- Bausteine ----------------
    def _demod(self, h):
        """Hologramm (N,N) -> komplexes Objektfeld (M,M); alles complex64."""
        S = xfft.fft2(xp.asarray(h, dtype=xp.float32) * self.win)
        return xfft.ifft2(S[self.ix] * self.sbwin)

    @staticmethod
    def _unit_phasor(z):
        """exp(-i*arg(z)) ohne Host-Sync, ohne cos/sin: z/|z|, konjugiert."""
        return xp.conj(z) / xp.abs(z)

    def _H_asm(self, z):
        if z not in self._H:
            f = xp.asarray(np.fft.fftfreq(self.m, self.dx_c), dtype=xp.float32)
            fx, fy = f[None, :], f[:, None]
            l = self.lam
            arg = 1 - (l * fx) ** 2 - (l * fy) ** 2
            H = xp.exp(1j * 2 * np.pi * z / l * xp.sqrt(xp.maximum(arg, 0)))
            df = 1.0 / (self.m * self.dx_c)
            f_lim = 1.0 / (l * np.sqrt((2 * df * z) ** 2 + 1))      # bandlimitiertes ASM
            H = H * (arg > 0) * (xp.abs(fx) < f_lim) * (xp.abs(fy) < f_lim)
            self._H[z] = H.astype(xp.complex64)
        return self._H[z]

    # ---------------- Hintergrund (einmalig / selten) ----------------
    def set_background(self, holograms):
        fields = [self._demod(h) for h in holograms]
        ref = fields[0]
        acc = xp.zeros_like(ref)
        for U in fields:                                   # globale Phase (Piston) angleichen
            acc += U * self._unit_phasor(xp.sum(U * xp.conj(ref)))
        self.Ub = acc / len(fields)
        p = xp.abs(self.Ub) ** 2
        self.Ub_inv = xp.conj(self.Ub) / (p + self.eps_rel * p.max())   # Wiener-artig, vorab berechnet
        self.reset()

    # ---------------- Objekt-Frames (Hot Loop) ----------------
    def reset(self):
        self.acc, self.count, self._ring, self._i = None, 0, None, 0

    def push(self, h):
        U = self._demod(h)
        U = U * self._unit_phasor(xp.sum(U * xp.conj(self.Ub) * self.ref_w))   # Piston -> 0
        if self.acc is None:
            self.acc = xp.zeros_like(U)
            if self.mode == "sliding":
                self._ring = xp.zeros((self.k,) + U.shape, U.dtype)
        if self.mode == "ema":
            self.acc = U if self.count == 0 else self.acc + self.alpha * (U - self.acc)
        elif self.mode == "sliding":
            self.acc += U - self._ring[self._i]            # Summe: neu rein, ältestes raus
            self._ring[self._i] = U
            self._i = (self._i + 1) % self.k
        else:
            self.acc += U
        self.count += 1

    def result(self, z_um=0.0):
        """Komplexes, hintergrundkompensiertes (und propagiertes) Objektfeld."""
        if self.mode == "ema":
            avg = self.acc
        else:
            n_eff = min(self.count, self.k) if self.mode == "sliding" else self.count
            avg = self.acc / n_eff
        R = avg * self.Ub_inv
        if z_um:
            R = xfft.ifft2(xfft.fft2(R) * self._H_asm(z_um))
        return R


# ---------------------------------------------------------------- Selbsttest
def _selftest():
    rng = np.random.default_rng(1)
    N, M = 256, 64
    k0 = (0.25, 0.18)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    g = lambda cx, cy, s: np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    phi = 1.5 * g(100, 125, 15) + 1.0 * g(165, 150, 10)
    env = 0.6 + 0.4 * g(128, 128, 100)
    sysph = 2 * np.pi * 3 * (1.2 * ((xx - 128) / N) ** 2 + 0.8 * ((yy - 128) / N) ** 2)
    Rw = np.exp(1j * 2 * np.pi * (k0[0] * xx + k0[1] * yy))

    def holo(obj, piston=0.0, noise=0.0):
        O = env * np.exp(1j * (sysph + (phi if obj else 0) + piston))
        h = np.abs(Rw + O) ** 2
        return (h + rng.normal(0, noise, h.shape) if noise else h).astype(np.float32)

    mask = np.ones((M, M), np.float32)
    sub = ((xx[::4, ::4] - 100) ** 2 + (yy[::4, ::4] - 125) ** 2 > 45 ** 2) & \
          ((xx[::4, ::4] - 165) ** 2 + (yy[::4, ::4] - 150) ** 2 > 35 ** 2)
    mask = (sub & (xx[::4, ::4] > 32) & (xx[::4, ::4] < 224) & (yy[::4, ::4] > 32) & (yy[::4, ::4] < 224)).astype(np.float32)
    kw = dict(n=N, crop=M, carrier=k0, pixel_size_um=0.5, wavelength_um=0.532, ref_mask=mask)

    truth = OffAxisPipeline(**kw)
    truth.set_background([holo(False)]); truth.push(holo(True))
    R0 = truth.result()

    def run(mode, nframes, **extra):
        p = OffAxisPipeline(**kw, mode=mode, **extra)
        p.set_background([holo(False, rng.normal(0, 0.6), 0.08) for _ in range(100)])
        for _ in range(nframes):
            p.push(holo(True, rng.normal(0, 0.6), 0.08))
        d = xp.angle(p.result() * xp.conj(R0))[16:48, 16:48]
        return float(xp.sqrt(xp.mean(d ** 2)))

    print(f"Backend: {'CuPy/GPU' if GPU else 'NumPy/CPU'}")
    e1 = run("block", 1); e32 = run("block", 32)
    print(f"block   N=1 : {e1:.4f} rad | N=32: {e32:.4f} rad (Faktor {e1/e32:.1f}, Soll ~5.7 bis Hintergrund-Floor)")
    print(f"sliding K=16, 40 Frames: {run('sliding', 40, window=16):.4f} rad")
    print(f"ema alpha=0.1, 60 Frames: {run('ema', 60, ema_alpha=0.1):.4f} rad")
    p = OffAxisPipeline(**kw); p.set_background([holo(False)]); p.push(holo(True))
    Rz = p.result(120.0)
    back = xfft.ifft2(xfft.fft2(Rz) * p._H_asm(-120.0))
    print(f"ASM +120um/-120um Round-trip: {float(xp.abs(back - p.result()).max()):.2e}")


if __name__ == "__main__":
    _selftest()
