"""Procedural planet height field (numpy).

`height(dirs)` takes unit direction vectors from the planet centre and returns
terrain elevation in metres relative to sea level. The same function feeds the
renderer and the flight physics, so what you see is what you land on.
"""

import numpy as np

PLANET_RADIUS = 20000.0      # sea level radius, metres
SEED = 7

_rng = np.random.default_rng(SEED)
_perm = _rng.permutation(256).astype(np.int32)
_P = np.concatenate([_perm, _perm])
# 16 gradient directions: the 12 cube-edge midpoints plus 4 repeats (classic improved Perlin)
_G = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 1, 0], [0, -1, 1], [-1, 1, 0], [0, -1, -1],
], dtype=np.float64)
# Gradient components indexed directly by the final hash (saves a lookup per corner)
_GX, _GY, _GZ = (_G[_P & 15, k].copy() for k in range(3))


def perlin(p):
    """Improved Perlin noise for an (N, 3) array of points, result roughly in [-1, 1]."""
    fl = np.floor(p)
    f = p - fl
    i = fl.astype(np.int64) & 255
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    xi, yi, zi = i[:, 0], i[:, 1], i[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    a = _P[xi] + yi
    b = _P[xi + 1] + yi
    aa, ab = _P[a] + zi, _P[a + 1] + zi
    ba, bb = _P[b] + zi, _P[b + 1] + zi

    def g(h, x, y, z):
        return _GX[h] * x + _GY[h] * y + _GZ[h] * z

    fx1, fy1, fz1 = fx - 1.0, fy - 1.0, fz - 1.0
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    x1 = g(aa, fx, fy, fz)
    x2 = g(ba, fx1, fy, fz)
    y1 = x1 + ux * (x2 - x1)
    x1 = g(ab, fx, fy1, fz)
    x2 = g(bb, fx1, fy1, fz)
    y2 = x1 + ux * (x2 - x1)
    z1 = y1 + uy * (y2 - y1)
    x1 = g(aa + 1, fx, fy, fz1)
    x2 = g(ba + 1, fx1, fy, fz1)
    y1 = x1 + ux * (x2 - x1)
    x1 = g(ab + 1, fx, fy1, fz1)
    x2 = g(bb + 1, fx1, fy1, fz1)
    y2 = x1 + ux * (x2 - x1)
    z2 = y1 + uy * (y2 - y1)
    return z1 + uz * (z2 - z1)


# Fixed per-octave offsets/rotations so octaves don't line up on the lattice
_OFFSETS = _rng.uniform(-1000, 1000, size=(64, 3))


def fbm(p, freq, octaves, gain=0.5, lacunarity=2.03, salt=0):
    total = np.zeros(len(p))
    amp = 1.0
    for o in range(octaves):
        total += amp * perlin(p * freq + _OFFSETS[(o + salt) % 64])
        freq *= lacunarity
        amp *= gain
    return total


def ridged(p, freq, octaves, salt=0):
    """Ridged multifractal: sharp crests, smooth valleys. Result roughly in [0, 1]."""
    total = np.zeros(len(p))
    amp, weight, norm = 1.0, np.ones(len(p)), 0.0
    for o in range(octaves):
        n = 1.0 - np.abs(perlin(p * freq + _OFFSETS[(o + salt) % 64]))
        n = n * n * weight
        weight = np.clip(n * 1.6, 0.0, 1.0)
        total += n * amp
        norm += amp
        freq *= 2.1
        amp *= 0.5
    return total / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _octaves_for(base_freq, min_wavelength, cap=14):
    """How many octaves until wavelength (in metres on the surface) drops below min_wavelength."""
    n, wl = 0, 2 * np.pi * PLANET_RADIUS / base_freq
    while wl > min_wavelength and n < cap:
        n += 1
        wl /= 2.0
    return max(n, 1)


# Landing pad flattening (set once the pad site has been chosen)
_pad = {"dir": None, "h": 0.0, "r0": 45.0, "r1": 130.0}


def set_pad(direction, h):
    _pad["dir"] = np.asarray(direction, dtype=np.float64)
    _pad["h"] = float(h)


def height(dirs, min_wavelength=4.0):
    """Elevation (m, relative to sea level) for (N, 3) unit vectors.

    min_wavelength drops noise octaves finer than the mesh can show, which keeps
    far-away low-detail chunks cheap and free of aliasing.
    """
    p = np.asarray(dirs, dtype=np.float64)
    mw = min_wavelength

    # Continents & ocean basins
    cont = fbm(p, 1.3, _octaves_for(1.3, max(mw, 900.0)), salt=0) * 1.5 + 0.10
    land = smoothstep(-0.02, 0.18, cont)
    h = np.where(cont > 0, cont * 600.0, cont * 3000.0)

    # Mountain ranges: ridged noise, only well inland, with a regional mask
    region = fbm(p, 2.2, 3, salt=10)
    mmask = smoothstep(0.08, 0.40, cont + 0.45 * region) * land
    ridge = ridged(p, 3.4, _octaves_for(3.4, mw * 2, 10), salt=20)
    h += ridge * ridge * 1900.0 * mmask

    # Rolling hills and small detail everywhere on land
    h += fbm(p, 14.0, _octaves_for(14.0, mw * 2, 8), salt=30) * 180.0 * land
    h += fbm(p, 90.0, _octaves_for(90.0, mw * 2, 6), salt=40) * 25.0 * (0.3 + land)

    # Lakes: inland depressions away from the mountains that dip under sea level
    lk = fbm(p, 6.0, 3, salt=50)
    lake = smoothstep(0.10, 0.22, lk) * smoothstep(0.02, 0.10, cont) * (1.0 - smoothstep(0.25, 0.6, mmask))
    h = h * (1.0 - lake) + (-35.0 + 10.0 * lk) * lake

    # Flatten the landing pad site
    if _pad["dir"] is not None:
        d = np.linalg.norm(p - _pad["dir"], axis=1) * PLANET_RADIUS
        w = 1.0 - smoothstep(_pad["r0"], _pad["r1"], d)
        h = h * (1.0 - w) + _pad["h"] * w
    return h


def height_at(direction):
    """Convenience: elevation for a single unit vector (full detail)."""
    return float(height(np.asarray(direction, dtype=np.float64).reshape(1, 3), 1.5)[0])


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def find_pad_site():
    """Pick a flat, low-ish spot on land that's close to the sea and mountains (scenic)."""
    pts = fibonacci_sphere(40000)          # ~180 m apart
    h = height(pts, 60.0)
    eps = 150.0 / PLANET_RADIUS
    # slope via finite differences along two tangents
    t1 = np.cross(pts, [0.0, 0.0, 1.0])
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9
    t2 = np.cross(pts, t1)
    hx = height(pts + t1 * eps, 60.0)
    hy = height(pts + t2 * eps, 60.0)
    slope = np.hypot(hx - h, hy - h) / 150.0
    ok = (h > 30) & (h < 350) & (slope < 0.05) & (np.abs(pts[:, 2]) < 0.8)
    idx = np.nonzero(ok)[0]
    sea = pts[h < -5]
    mtn = pts[h > 1200]
    best, best_score = None, -1e9
    rng = np.random.default_rng(3)
    for k in rng.permutation(idx)[:600]:
        p = pts[k]
        ds = np.min(np.linalg.norm(sea - p, axis=1)) * PLANET_RADIUS if len(sea) else 1e9
        dm = np.min(np.linalg.norm(mtn - p, axis=1)) * PLANET_RADIUS if len(mtn) else 1e9
        score = -abs(ds - 1500) - 0.5 * abs(dm - 5000) - slope[k] * 20000
        if score > best_score:
            best, best_score = p, score
    return best / np.linalg.norm(best)
