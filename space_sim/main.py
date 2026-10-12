"""A small space sim built with Panda3D.

Start parked on a landing pad on a spherical planet, take off, fly over
mountains, valleys, seas and lakes, climb out of the atmosphere into space,
then turn around and land anywhere you like.

The planet (terrain, ocean, atmosphere), the ship and the pad are all
generated procedurally at startup.
Run with:  .venv/bin/python main.py
"""

import math
import queue
import sys
import threading
import time

import numpy as np

from panda3d.core import (
    loadPrcFileData,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    KeyboardButton,
    NodePath,
    Point3,
    Quat,
    Shader,
    TransparencyAttrib,
    Vec3,
    Vec4,
    WindowProperties,
    BoundingSphere,
)

import terrain
from terrain import PLANET_RADIUS as R

SCREENSHOT_MODE = "--screenshot" in sys.argv
sys.setswitchinterval(0.001)   # keep the main thread responsive while the terrain worker runs

loadPrcFileData("", """
window-title Panda3D space sim
win-size 1280 720
sync-video true
show-frame-rate-meter true
framebuffer-multisample 1
multisamples 4
gl-version 3 2
""")
if SCREENSHOT_MODE:
    loadPrcFileData("", "window-type offscreen\nsync-video false\nshow-frame-rate-meter false")

from direct.showbase.ShowBase import ShowBase  # noqa: E402  (after prc config)
from direct.gui.OnscreenText import OnscreenText  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
CAM_FAR = 3.0e6
ATMO_TOP = 14000.0           # atmosphere shell height above sea level
SPACE_ALT = 12000.0          # where the HUD says we're "in space"
G0 = 9.81                    # surface gravity

CHUNK_N = 33                 # vertices per chunk side
MAX_DEPTH = 9                # quadtree depth: ~2 m vertex spacing at the finest level
SPLIT_K = 1.7                # split when closer than K * chunk size
BUILD_BUDGET = 0.004         # seconds per frame spent turning chunk data into Geoms

# Ship performance (accelerations in m/s^2, speeds in m/s)
MAIN_ACC = 30.0
BOOST_ACC = 80.0
RETRO_ACC = 15.0
VERT_ACC = 22.0
DOWN_ACC = 12.0
LAT_ACC = 14.0
DRAG_K = 4.8e-4              # drag = K * rho * v^2, rho = 1 at sea level
DRAG_H = 2000.0
FA_MAX_SPEED = 320.0
FA_BOOST_SPEED = 1500.0
FA_VERT_SPEED = 40.0
FA_LAT_SPEED = 50.0
PITCH_RATE, YAW_RATE, ROLL_RATE = 70.0, 50.0, 110.0   # deg/s for keyboard
MOUSE_SENS = 0.12
LEG_Z = -2.3                 # bottom of the landing feet in ship space

# ---------------------------------------------------------------------------
# Shaders. Everything uses a logarithmic depth buffer so we can see from the
# landing pad's bolts to the far side of the planet without z-fighting.
# ---------------------------------------------------------------------------
ATMO_GLSL = """
const float PR = %(R).1f;
const float RA = %(RA).1f;
const float HR = 2000.0;
const float HM = 500.0;
const vec3 BETA_R = vec3(5.8e-6, 13.5e-6, 33.1e-6) * 3.5;
const vec3 BETA_M = vec3(2.1e-5) * 1.5;
const float SUN_I = 22.0;
uniform vec3 sunDir;

float airmass(float mu) {
    float ang = min(degrees(acos(clamp(mu, -1.0, 1.0))), 93.0);
    return 1.0 / (max(mu, 0.0) + 0.15 * pow(93.885 - ang, -1.253));
}

// Transmittance of sunlight reaching a point at height h with local sun cosine mu
vec3 sunTransmit(float h, float mu) {
    float m = airmass(mu);
    vec3 tau = BETA_R * HR * exp(-h / HR) * m + BETA_M * 1.1 * HM * exp(-h / HM) * m;
    return exp(-tau) * smoothstep(-0.12, 0.04, mu);
}

// Single scattering along a ray. Returns in-scattered light, sets transmittance.
vec3 atmosphere(vec3 ro, vec3 rd, float tmax, out vec3 trans) {
    trans = vec3(1.0);
    float b = dot(ro, rd);
    float c = dot(ro, ro) - RA * RA;
    float disc = b * b - c;
    if (disc <= 0.0) return vec3(0.0);
    float s = sqrt(disc);
    float t0 = max(-b - s, 0.0);
    float t1 = min(-b + s, tmax);
    if (t1 <= t0) return vec3(0.0);
    const int STEPS = 10;
    float dt = (t1 - t0) / float(STEPS);
    vec3 sumR = vec3(0.0), sumM = vec3(0.0);
    float odR = 0.0, odM = 0.0;
    for (int i = 0; i < STEPS; ++i) {
        vec3 p = ro + rd * (t0 + (float(i) + 0.5) * dt);
        float r = length(p);
        float h = max(r - PR, 0.0);
        float dR = exp(-h / HR) * dt;
        float dM = exp(-h / HM) * dt;
        odR += 0.5 * dR; odM += 0.5 * dM;
        vec3 att = exp(-(BETA_R * odR + BETA_M * 1.1 * odM)) * sunTransmit(h, dot(p / r, sunDir));
        odR += 0.5 * dR; odM += 0.5 * dM;
        sumR += att * dR;
        sumM += att * dM;
    }
    trans = exp(-(BETA_R * odR + BETA_M * 1.1 * odM));
    float mu = dot(rd, sunDir);
    float pR = 3.0 / (16.0 * 3.14159) * (1.0 + mu * mu);
    const float g = 0.76;
    float pM = 3.0 / (8.0 * 3.14159) * ((1.0 - g * g) * (1.0 + mu * mu)) /
               ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * mu, 1.5));
    return SUN_I * (sumR * BETA_R * pR + sumM * BETA_M * pM);
}

vec3 tonemap(vec3 c) {
    c = vec3(1.0) - exp(-c * 1.1);
    return pow(c, vec3(1.0 / 1.15));
}
""" % {"R": R, "RA": R + ATMO_TOP}

LOGZ_VS = """
uniform float logFC;
out float flogz;
void logDepth() {
    gl_Position.z = (log2(max(1e-6, 1.0 + gl_Position.w)) * logFC - 1.0) * gl_Position.w;
    flogz = 1.0 + gl_Position.w;
}
"""
LOGZ_FS = """
uniform float logFC;
in float flogz;
void logDepth() { gl_FragDepth = log2(flogz) * logFC * 0.5; }
"""

LIT_VS = """#version 150
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
out vec3 wpos;
out vec3 wnormal;
out vec4 vcolor;
""" + LOGZ_VS + """
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    logDepth();
    wpos = (p3d_ModelMatrix * p3d_Vertex).xyz;
    wnormal = mat3(p3d_ModelMatrix) * p3d_Normal;
    vcolor = p3d_Color;
}
"""

# Terrain, ship and pad: vertex colour albedo lit by the (atmosphere-filtered) sun
LIT_FS = """#version 150
in vec3 wpos;
in vec3 wnormal;
in vec4 vcolor;
uniform vec3 camPos;
uniform float emissive;
uniform vec4 tint;
out vec4 fragColor;
""" + LOGZ_FS + ATMO_GLSL + """
void main() {
    logDepth();
    vec3 albedo = vcolor.rgb * tint.rgb;
    vec3 n = normalize(wnormal);
    float r = length(wpos);
    vec3 up = wpos / r;
    float h = max(r - PR, 0.0);
    float muUp = dot(up, sunDir);
    vec3 sunCol = sunTransmit(h, muUp) * 3.2;
    float day = smoothstep(-0.25, 0.3, muUp);
    vec3 sky = vec3(0.45, 0.6, 0.9) * (0.08 + 0.6 * day) * (0.6 + 0.4 * dot(n, up));
    vec3 lit = albedo * (sunCol * max(dot(n, sunDir), 0.0) + sky + vec3(0.015));
    lit = mix(lit, albedo * 4.0, emissive);
    vec3 v = wpos - camPos;
    float dist = length(v);
    vec3 trans;
    vec3 ins = atmosphere(camPos, v / dist, dist, trans);
    fragColor = vec4(tonemap(lit * trans + ins), tint.a);
}
"""

# Ocean: depth-tinted water with sun glint and sky reflection
OCEAN_FS = """#version 150
in vec3 wpos;
in vec3 wnormal;
in vec4 vcolor;
uniform vec3 camPos;
uniform float osg_FrameTime;
out vec4 fragColor;
""" + LOGZ_FS + ATMO_GLSL + """
void main() {
    logDepth();
    vec3 up = normalize(wnormal);
    vec3 v = wpos - camPos;
    float dist = length(v);
    vec3 vd = v / dist;
    // cheap animated ripples, fading out with distance
    float t = osg_FrameTime;
    vec3 q = wpos * 0.15;
    vec3 w = vec3(sin(q.x * 1.3 + t * 1.1) + sin(q.y * 2.1 - t * 0.9),
                  sin(q.y * 1.7 + t * 1.3) + sin(q.z * 2.3 + t * 0.7),
                  sin(q.z * 1.1 - t * 1.2) + sin(q.x * 1.9 + t));
    vec3 n = normalize(up + (w - up * dot(w, up)) * 0.03 * exp(-dist / 600.0));
    float depth = vcolor.r * 200.0;
    vec3 shallow = vec3(0.10, 0.42, 0.45);
    vec3 deep = vec3(0.01, 0.07, 0.20);
    vec3 water = mix(shallow, deep, smoothstep(0.0, 60.0, depth));
    float muUp = dot(up, sunDir);
    vec3 sunCol = sunTransmit(0.0, muUp) * 3.2;
    float day = smoothstep(-0.25, 0.3, muUp);
    vec3 body = water * (sunCol * max(muUp, 0.0) * 0.7 + vec3(0.35, 0.55, 1.0) * 0.4 * day + 0.01);
    float fres = 0.02 + 0.98 * pow(1.0 - max(dot(-vd, n), 0.0), 5.0);
    vec3 skyRefl = vec3(0.45, 0.65, 1.0) * day * 1.2;
    vec3 refl = reflect(vd, n);
    float spec = pow(max(dot(refl, sunDir), 0.0), 400.0) * 60.0;
    vec3 col = mix(body, skyRefl, fres) + sunCol * spec * smoothstep(0.0, 0.1, muUp);
    // foam near the shore
    col = mix(col, vec3(0.8) * (sunCol * max(muUp, 0.0) + 0.2), (1.0 - smoothstep(0.0, 1.5, depth)) * 0.5);
    vec3 trans;
    vec3 ins = atmosphere(camPos, vd, dist, trans);
    fragColor = vec4(tonemap(col * trans + ins), 1.0);
}
"""

SKY_VS = """#version 150
in vec4 p3d_Vertex;
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
out vec3 wpos;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    wpos = (p3d_ModelMatrix * p3d_Vertex).xyz;
}
"""

SKY_FS = """#version 150
in vec3 wpos;
uniform vec3 camPos;
out vec4 fragColor;
""" + ATMO_GLSL + """
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
void main() {
    vec3 rd = normalize(wpos - camPos);
    // where does the view ray hit the planet (sea level sphere)?
    float b = dot(camPos, rd);
    float c = dot(camPos, camPos) - PR * PR;
    float disc = b * b - c;
    float tmax = 1e9;
    bool hitPlanet = false;
    if (disc > 0.0) {
        float t = -b - sqrt(disc);
        if (t > 0.0) { tmax = t; hitPlanet = true; }
    }
    vec3 trans;
    vec3 col = atmosphere(camPos, rd, tmax, trans);
    if (!hitPlanet) {
        float sd = dot(rd, sunDir);
        col += trans * vec3(1.0, 0.95, 0.85) * (smoothstep(0.99985, 0.99995, sd) * 60.0
                                                 + pow(max(sd, 0.0), 2000.0) * 6.0);
        // procedural star field
        vec3 p = rd * 220.0;
        vec3 cell = floor(p);
        vec3 f = fract(p);
        float hsh = hash(cell);
        vec3 sp = vec3(hash(cell + 3.1), hash(cell + 7.7), hash(cell + 13.3)) * 0.6 + 0.2;
        float star = step(0.93, hsh) * smoothstep(0.09, 0.0, length(f - sp)) * (hsh - 0.93) * 40.0;
        float skyLum = dot(col, vec3(0.3, 0.5, 0.2));
        col += vec3(star) * trans * max(0.0, 1.0 - skyLum * 4.0);
    }
    fragColor = vec4(tonemap(col), 1.0);
}
"""

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
_arr_fmt = GeomVertexArrayFormat()
_arr_fmt.addColumn("vertex", 3, Geom.NT_float32, Geom.C_point)
_arr_fmt.addColumn("normal", 3, Geom.NT_float32, Geom.C_normal)
_arr_fmt.addColumn("color", 4, Geom.NT_float32, Geom.C_color)
VFORMAT = GeomVertexFormat.registerFormat(GeomVertexFormat(_arr_fmt))


def make_triangles(indices):
    prim = GeomTriangles(Geom.UHStatic)
    big = indices.max() > 65535
    prim.setIndexType(Geom.NT_uint32 if big else Geom.NT_uint16)
    arr = prim.modifyVertices()
    arr.uncleanSetNumRows(len(indices))
    memoryview(arr).cast("B")[:] = indices.astype(np.uint32 if big else np.uint16).tobytes()
    return prim


def make_geom_node(name, verts, prim):
    """verts: (n, 10) float32 array = position, normal, rgba."""
    vdata = GeomVertexData(name, VFORMAT, Geom.UHStatic)
    vdata.uncleanSetNumRows(len(verts))
    memoryview(vdata.modifyArray(0)).cast("B")[:] = np.ascontiguousarray(verts, dtype=np.float32).tobytes()
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


class MeshBuilder:
    """Accumulates flat-shaded, vertex-coloured triangles for the procedural models."""

    def __init__(self):
        self.rows = []

    def tri(self, a, b, c, color):
        a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 0 else np.array([0, 0, 1.0])
        for p in (a, b, c):
            self.rows.append([*p, *n, *color])

    def quad(self, a, b, c, d, color):
        self.tri(a, b, c, color)
        self.tri(a, c, d, color)

    def box(self, x0, x1, y0, y1, z0, z1, color, nose=1.0, nose_z=1.0):
        """Axis-aligned box; the +Y face can be shrunk (nose) for wedge shapes."""
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2

        def front(x, z):
            return (cx + (x - cx) * nose, y1, cz + (z - cz) * nose_z)

        p = [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1),
             front(x0, z0), front(x1, z0), front(x1, z1), front(x0, z1)]
        self.quad(p[0], p[1], p[2], p[3], color)      # back  (-Y)
        self.quad(p[5], p[4], p[7], p[6], color)      # front (+Y)
        self.quad(p[4], p[0], p[3], p[7], color)      # left  (-X)
        self.quad(p[1], p[5], p[6], p[2], color)      # right (+X)
        self.quad(p[3], p[2], p[6], p[7], color)      # top
        self.quad(p[4], p[5], p[1], p[0], color)      # bottom

    def cone(self, base_center, radius, length, color, segs=12):
        """Cone along -Y from base_center (used for engine flames)."""
        bx, by, bz = base_center
        tip = (bx, by - length, bz)
        for i in range(segs):
            a0 = 2 * math.pi * i / segs
            a1 = 2 * math.pi * (i + 1) / segs
            p0 = (bx + math.cos(a0) * radius, by, bz + math.sin(a0) * radius)
            p1 = (bx + math.cos(a1) * radius, by, bz + math.sin(a1) * radius)
            self.tri(p1, p0, tip, color)
            self.tri(p0, p1, base_center, color)

    def disc(self, center, radius, z, color, segs=48):
        cx, cy = center
        for i in range(segs):
            a0 = 2 * math.pi * i / segs
            a1 = 2 * math.pi * (i + 1) / segs
            self.tri((cx, cy, z), (cx + math.cos(a0) * radius, cy + math.sin(a0) * radius, z),
                     (cx + math.cos(a1) * radius, cy + math.sin(a1) * radius, z), color)

    def node(self, name):
        verts = np.array(self.rows, dtype=np.float32)
        return make_geom_node(name, verts, make_triangles(np.arange(len(verts))))


# ---------------------------------------------------------------------------
# Planet: cube-sphere quadtree of terrain chunks
# ---------------------------------------------------------------------------
# (normal, u, v) per cube face with u x v = normal so triangles wind outward
FACES = [
    (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    (np.array([-1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0])),
    (np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])),
    (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
    (np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])),
]


def spherify(c):
    """Cube point -> unit sphere with the area-preserving-ish 'spherified cube' mapping."""
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    s = np.stack([
        x * np.sqrt(np.maximum(1 - y2 / 2 - z2 / 2 + y2 * z2 / 3, 1e-9)),
        y * np.sqrt(np.maximum(1 - z2 / 2 - x2 / 2 + z2 * x2 / 3, 1e-9)),
        z * np.sqrt(np.maximum(1 - x2 / 2 - y2 / 2 + x2 * y2 / 3, 1e-9)),
    ], axis=-1)
    return s / np.linalg.norm(s, axis=-1, keepdims=True)


def _grid_indices(n):
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    k = (i * n + j).ravel()
    return np.stack([k, k + n, k + n + 1, k, k + n + 1, k + 1], axis=1).ravel()


def _border_loop(n):
    top = [(0, j) for j in range(n)]
    right = [(i, n - 1) for i in range(1, n)]
    bottom = [(n - 1, j) for j in range(n - 2, -1, -1)]
    left = [(i, 0) for i in range(n - 2, 0, -1)]
    return np.array([i * n + j for i, j in top + right + bottom + left])


BORDER = _border_loop(CHUNK_N)


def _terrain_indices(n):
    grid = _grid_indices(n)
    nb = len(BORDER)
    base = n * n
    tris = []
    for s in range(nb):
        a, b = BORDER[s], BORDER[(s + 1) % nb]
        sa, sb = base + s, base + (s + 1) % nb
        tris += [a, b, sb, a, sb, sa, a, sb, b, a, sa, sb]   # both windings
    return np.concatenate([grid, np.array(tris)])


TERRAIN_PRIM_IDX = _terrain_indices(CHUNK_N)
OCEAN_PRIM_IDX = _grid_indices(CHUNK_N)

C_SAND = np.array([0.80, 0.74, 0.52])
C_SEABED = np.array([0.55, 0.50, 0.36])
C_GRASS = np.array([0.30, 0.47, 0.15])
C_FOREST = np.array([0.12, 0.27, 0.09])
C_DRY = np.array([0.50, 0.46, 0.30])
C_ROCK = np.array([0.40, 0.37, 0.34])
C_DARKROCK = np.array([0.28, 0.26, 0.25])
C_SNOW = np.array([0.95, 0.96, 1.0])


def terrain_colors(h, flat, dirs):
    """Albedo per vertex from height, flatness (normal . up) and position."""
    n = len(h)
    veg = terrain.perlin(dirs * 160.0 + 11.0)            # forest vs meadow patches
    var = terrain.perlin(dirs * 1400.0 + 5.0) * 0.5      # small-scale colour variation
    t = lambda a: a[:, None]                             # noqa: E731
    col = np.broadcast_to(C_GRASS, (n, 3)).copy()
    col = col + (C_FOREST - col) * t(terrain.smoothstep(-0.05, 0.25, veg + var * 0.3)
                                       * (1 - terrain.smoothstep(700, 1100, h)))
    col = col + (C_DRY - col) * t(terrain.smoothstep(800, 1300, h + var * 300))
    col = col + (C_SAND - col) * t(1 - terrain.smoothstep(2.0, 9.0, h + var * 6))
    col = col + (C_SEABED - col) * t(1 - terrain.smoothstep(-6.0, -1.0, h))
    rock = np.maximum(1 - terrain.smoothstep(0.70, 0.86, flat + var * 0.08),
                      terrain.smoothstep(1500, 2000, h + var * 400))
    rockc = C_ROCK + (C_DARKROCK - C_ROCK) * t(np.clip(var + 0.5, 0, 1))
    col = col + (rockc - col) * t(rock * terrain.smoothstep(-2.0, 5.0, h))
    snowline = 2300 - 1100 * np.abs(dirs[:, 2])
    snow = terrain.smoothstep(snowline - 150, snowline + 150, h + var * 500) * terrain.smoothstep(0.62, 0.78, flat)
    col = col + (C_SNOW - col) * t(snow)
    col *= t(1.0 + var * 0.25)
    return col


def build_chunk(face, a0, b0, size):
    """Pure numpy: compute vertex arrays for one chunk (safe to run in a worker thread)."""
    n = CHUNK_N
    ds = size / (n - 1)
    idx = np.arange(n + 2) - 1.0
    A, B = np.meshgrid(a0 + idx * ds, b0 + idx * ds, indexing="ij")
    fn, fu, fv = FACES[face]
    cube = fn + A[..., None] * fu + B[..., None] * fv
    dirs = spherify(cube)
    spacing = ds * R * math.pi / 4
    h = terrain.height(dirs.reshape(-1, 3), max(2.0 * spacing, 3.0)).reshape(n + 2, n + 2)
    pos = dirs * (R + h)[..., None]
    dA = pos[2:, 1:-1] - pos[:-2, 1:-1]
    dB = pos[1:-1, 2:] - pos[1:-1, :-2]
    nrm = np.cross(dA, dB).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    P = pos[1:-1, 1:-1].reshape(-1, 3)
    D = dirs[1:-1, 1:-1].reshape(-1, 3)
    H = h[1:-1, 1:-1].ravel()
    center_dir = D[(n * n) // 2]
    origin = center_dir * R
    flat = np.sum(nrm * D, axis=1)
    col = terrain_colors(H, flat, D)

    skirt = max(8.0, size * R * 0.02)
    sk_pos = P[BORDER] - D[BORDER] * skirt
    verts = np.zeros((n * n + len(BORDER), 10), dtype=np.float32)
    verts[:, 0:3] = np.concatenate([P, sk_pos]) - origin
    verts[:, 3:6] = np.concatenate([nrm, nrm[BORDER]])
    verts[:, 6:9] = np.concatenate([col, col[BORDER] * 0.8])
    verts[:, 9] = 1.0

    ocean = None
    if H.min() < 0.5:
        ocean = np.zeros((n * n, 10), dtype=np.float32)
        ocean[:, 0:3] = D * R - origin
        ocean[:, 3:6] = D
        ocean[:, 6] = np.clip(-H / 200.0, 0.0, 1.0)
        ocean[:, 9] = 1.0

    radii = np.linalg.norm(P, axis=1)
    cos_ang = np.clip(D @ center_dir, -1, 1)
    hc = float(H[(n * n) // 2])
    return {
        "verts": verts, "ocean": ocean, "origin": origin,
        "center": center_dir * (R + hc),
        "ctuple": tuple(float(a) for a in center_dir * (R + hc)),
        "cdir": tuple(float(a) for a in center_dir), "center_dir": center_dir,
        "r_top": float(max(radii.max(), R)),
        "ang": float(np.arccos(cos_ang.min())),
        "bound": float(np.max(np.linalg.norm(np.concatenate([P, D * R]) - center_dir * (R + hc), axis=1))),
    }


class Chunk:
    __slots__ = ("face", "a0", "b0", "size", "depth", "children", "np", "data",
                 "disposed", "size_m", "parent")

    def __init__(self, face, a0, b0, size, depth, parent=None):
        self.face, self.a0, self.b0, self.size, self.depth = face, a0, b0, size, depth
        self.children = None
        self.np = None
        self.data = None
        self.disposed = False
        self.parent = parent
        self.size_m = size * R * math.pi / 4

    def make_children(self):
        h = self.size / 2
        d = self.depth + 1
        return [Chunk(self.face, self.a0, self.b0, h, d, self),
                Chunk(self.face, self.a0 + h, self.b0, h, d, self),
                Chunk(self.face, self.a0, self.b0 + h, h, d, self),
                Chunk(self.face, self.a0 + h, self.b0 + h, h, d, self)]


class Planet:
    """Quadtree level-of-detail terrain + ocean. Chunk meshes are computed on a
    worker thread; the main thread only wraps finished arrays into Geoms."""

    def __init__(self, parent):
        self.root = parent.attachNewNode("planet")
        self.terrain_prim = make_triangles(TERRAIN_PRIM_IDX)
        self.ocean_prim = make_triangles(OCEAN_PRIM_IDX)
        self.ocean_shader = Shader.make(Shader.SL_GLSL, LIT_VS, OCEAN_FS)
        self.roots = [Chunk(f, -1.0, -1.0, 2.0, 0) for f in range(6)]
        self.requests = []                      # chunk groups waiting for the worker
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.results = queue.Queue()
        self.cam = np.zeros(3)
        self.n_chunks = 0
        for c in self.roots:
            self._attach(c, build_chunk(c.face, c.a0, c.b0, c.size))
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    # -- worker -------------------------------------------------------------
    def _priority(self, group):
        p = group[0].parent
        d = np.linalg.norm(p.data["center"] - self.cam) - p.data["bound"]
        return max(d, 0.0) / p.size_m

    def _work(self):
        while True:
            self.wake.wait()
            with self.lock:
                live = [g for g in self.requests if not g[0].disposed]
                self.requests = live
                if not live:
                    self.wake.clear()
                    continue
                group = min(live, key=self._priority)
                self.requests.remove(group)
            if group[0].disposed:
                continue
            out = [build_chunk(c.face, c.a0, c.b0, c.size) for c in group]
            self.results.put((group, out))

    # -- main thread --------------------------------------------------------
    def _attach(self, chunk, data):
        chunk.data = data
        np_ = NodePath("chunk")
        np_.attachNewNode(make_geom_node("terrain", data["verts"], self.terrain_prim))
        if data["ocean"] is not None:
            o = np_.attachNewNode(make_geom_node("ocean", data["ocean"], self.ocean_prim))
            o.setShader(self.ocean_shader)
        np_.setPos(*data["origin"])
        np_.node().setBounds(BoundingSphere(Point3(*(data["center"] - data["origin"])), data["bound"]))
        np_.node().setFinal(True)
        chunk.np = np_
        self.n_chunks += 1

    def _dispose(self, chunk):
        chunk.disposed = True
        if chunk.np is not None:
            chunk.np.removeNode()
            chunk.np = None
            self.n_chunks -= 1
        if chunk.children:
            for c in chunk.children:
                self._dispose(c)
        chunk.children = None

    @staticmethod
    def _horizon_visible(d, cx, cy, cz, cam_r, occ):
        if cam_r <= occ:
            return True
        ux, uy, uz = d["cdir"]
        cos_a = (ux * cx + uy * cy + uz * cz) / cam_r
        ang = math.acos(max(-1.0, min(1.0, cos_a))) - d["ang"]
        limit = math.acos(occ / cam_r) + math.acos(min(1.0, occ / d["r_top"]))
        return ang < limit

    def update(self, cam_pos, sync=False):
        cam = np.array(cam_pos, dtype=np.float64)
        self.cam = cam
        cx, cy, cz = (float(a) for a in cam)
        cam_r = math.sqrt(cx * cx + cy * cy + cz * cz)
        occ = R - 150.0
        new_requests = []
        horizon = self._horizon_visible

        def visit(c):
            d = c.data
            vis = horizon(d, cx, cy, cz, cam_r, occ)
            px, py, pz = d["ctuple"]
            dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2 + (pz - cz) ** 2) - d["bound"]
            # a little extra detail on the big chunks keeps the planet round from orbit
            k = max(SPLIT_K, 3.2 - 0.5 * c.depth) * (1.25 if c.children else 1.0)
            want = vis and c.depth < MAX_DEPTH and dist < k * c.size_m
            if want:
                if c.children is None:
                    c.children = c.make_children()
                    new_requests.append(c.children)
                if all(ch.data is not None for ch in c.children):
                    c.np.detachNode()
                    for ch in c.children:
                        visit(ch)
                    return
            elif c.children is not None:
                for ch in c.children:
                    self._dispose(ch)
                c.children = None
            if vis:
                if not c.np.hasParent():
                    c.np.reparentTo(self.root)
            else:
                c.np.detachNode()

        for c in self.roots:
            visit(c)

        if sync:
            for g in new_requests:
                for ch in g:
                    self._attach(ch, build_chunk(ch.face, ch.a0, ch.b0, ch.size))
            return bool(new_requests)

        if new_requests:
            with self.lock:
                self.requests.extend(new_requests)
            self.wake.set()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < BUILD_BUDGET:
            try:
                group, out = self.results.get_nowait()
            except queue.Empty:
                break
            if group[0].disposed:
                continue
            for ch, data in zip(group, out):
                self._attach(ch, data)
        return False

    def pending(self):
        return len(self.requests) + self.results.qsize()


# ---------------------------------------------------------------------------
# Ship & pad models
# ---------------------------------------------------------------------------
HULL = (0.82, 0.83, 0.86, 1)
HULL_DARK = (0.30, 0.32, 0.36, 1)
STRIPE = (0.95, 0.45, 0.10, 1)
GLASS = (0.10, 0.20, 0.35, 1)


def build_ship(parent):
    ship = parent.attachNewNode("ship")
    mb = MeshBuilder()
    # fuselage with tapered nose
    mb.box(-1.3, 1.3, -5.0, 1.5, -0.9, 1.0, HULL)
    mb.box(-1.3, 1.3, 1.5, 6.5, -0.9, 1.0, HULL, nose=0.35, nose_z=0.4)
    mb.box(-0.9, 0.9, 0.0, 4.0, 1.0, 1.8, GLASS, nose=0.5, nose_z=0.3)
    mb.box(-1.32, 1.32, -2.0, -1.4, -0.5, 1.02, STRIPE)
    # wings
    for s in (-1, 1):
        x0, x1 = (1.3, 6.0) if s > 0 else (-6.0, -1.3)
        mb.box(x0, x1, -3.5, 0.5, -0.25, 0.15, HULL, nose=0.6)
        tip = 5.6 * s
        mb.box(tip - 0.4, tip + 0.4, -3.6, -1.2, 0.15, 1.4, STRIPE, nose=0.5)
        # engine pods
        ex = 2.0 * s
        mb.box(ex - 0.75, ex + 0.75, -6.0, -1.0, -0.8, 0.6, HULL_DARK, nose=0.7, nose_z=0.7)
        # landing legs
        mb.box(3.0 * s - 0.15, 3.0 * s + 0.15, -1.9, -1.6, LEG_Z + 0.2, -0.2, HULL_DARK)
        mb.box(3.0 * s - 0.5, 3.0 * s + 0.5, -2.2, -1.3, LEG_Z, LEG_Z + 0.2, HULL_DARK)
    mb.box(-0.15, 0.15, 3.6, 3.9, LEG_Z + 0.2, -0.7, HULL_DARK)
    mb.box(-0.5, 0.5, 3.3, 4.2, LEG_Z, LEG_Z + 0.2, HULL_DARK)
    ship.attachNewNode(mb.node("hull"))

    # emissive bits: nav lights and engine flames
    lights = MeshBuilder()
    lights.box(-6.1, -5.7, -0.4, 0.0, -0.1, 0.2, (1.0, 0.1, 0.1, 1))
    lights.box(5.7, 6.1, -0.4, 0.0, -0.1, 0.2, (0.1, 1.0, 0.2, 1))
    for s in (-1, 1):
        lights.box(2.0 * s - 0.5, 2.0 * s + 0.5, -6.05, -5.95, -0.6, 0.4, (1.0, 0.55, 0.2, 1))
    ln = ship.attachNewNode(lights.node("lights"))
    ln.setShaderInput("emissive", 1.0)

    flames = []
    for s in (-1, 1):
        fb = MeshBuilder()
        fb.cone((0, 0, 0), 0.55, 1.0, (1.0, 0.55, 0.15, 1))
        f = ship.attachNewNode(fb.node("flame"))
        f.setPos(2.0 * s, -6.0, -0.1)
        f.setShaderInput("emissive", 1.0)
        f.setShaderInput("tint", Vec4(1, 1, 1, 0.85))
        f.setTransparency(TransparencyAttrib.MAlpha)
        f.setDepthWrite(False)
        f.setBin("fixed", 10)
        flames.append(f)
    # downward hover jets (under the wings)
    jets = []
    for x, y in ((3.0, -0.6), (-3.0, -0.6), (0.0, 4.5)):
        fb = MeshBuilder()
        fb.cone((0, 0, 0), 0.35, 1.0, (0.5, 0.75, 1.0, 1))
        j = ship.attachNewNode(fb.node("jet"))
        j.setPos(x, y, -0.6)
        j.setP(-90)        # point -Y (cone axis) downward
        j.setShaderInput("emissive", 1.0)
        j.setShaderInput("tint", Vec4(1, 1, 1, 0.7))
        j.setTransparency(TransparencyAttrib.MAlpha)
        j.setDepthWrite(False)
        j.setBin("fixed", 11)
        jets.append(j)
    return ship, flames, jets


def build_pad(parent):
    pad = parent.attachNewNode("pad")
    mb = MeshBuilder()
    mb.disc((0, 0), 34.0, 0.0, (0.42, 0.42, 0.44, 1))
    mb.disc((0, 0), 30.0, 0.05, (0.30, 0.30, 0.32, 1))
    mb.disc((0, 0), 12.0, 0.10, (0.95, 0.75, 0.10, 1))
    mb.disc((0, 0), 10.5, 0.15, (0.30, 0.30, 0.32, 1))
    # big "H"
    y = (0.95, 0.95, 0.95, 1)
    mb.box(-5.0, -3.5, -6.0, 6.0, 0.15, 0.2, y)
    mb.box(3.5, 5.0, -6.0, 6.0, 0.15, 0.2, y)
    mb.box(-3.5, 3.5, -0.75, 0.75, 0.15, 0.2, y)
    # base skirt so the disc never floats above uneven ground
    for i in range(48):
        a0, a1 = 2 * math.pi * i / 48, 2 * math.pi * (i + 1) / 48
        p0 = (math.cos(a0) * 34, math.sin(a0) * 34)
        p1 = (math.cos(a1) * 34, math.sin(a1) * 34)
        mb.quad((p0[0], p0[1], -4), (p1[0], p1[1], -4), (p1[0], p1[1], 0), (p0[0], p0[1], 0), (0.35, 0.35, 0.36, 1))
    # control tower
    mb.box(42, 48, -4, 2, -3, 14, (0.75, 0.75, 0.78, 1))
    mb.box(41, 49, -5, 3, 14, 18, GLASS)
    mb.box(40.5, 49.5, -5.5, 3.5, 18, 18.6, (0.6, 0.6, 0.62, 1))
    pad.attachNewNode(mb.node("pad"))
    lights = MeshBuilder()
    for i in range(16):
        a = 2 * math.pi * i / 16
        x, yy = math.cos(a) * 32, math.sin(a) * 32
        lights.box(x - 0.3, x + 0.3, yy - 0.3, yy + 0.3, 0.0, 0.5, (0.2, 0.6, 1.0, 1))
    lights.box(44.7, 45.3, -1.3, -0.7, 18.6, 21, (1.0, 0.15, 0.1, 1))
    ln = pad.attachNewNode(lights.node("pad_lights"))
    ln.setShaderInput("emissive", 1.0)
    return pad


def build_sky_dome(parent, radius=1000.0, segs=24):
    mb = MeshBuilder()
    for i in range(segs):
        t0, t1 = math.pi * i / segs, math.pi * (i + 1) / segs
        for j in range(segs * 2):
            p0, p1 = 2 * math.pi * j / (segs * 2), 2 * math.pi * (j + 1) / (segs * 2)

            def pt(t, p):
                return (radius * math.sin(t) * math.cos(p), radius * math.sin(t) * math.sin(p), radius * math.cos(t))
            mb.quad(pt(t0, p0), pt(t0, p1), pt(t1, p1), pt(t1, p0), (1, 1, 1, 1))
    dome = parent.attachNewNode(mb.node("sky"))
    dome.setTwoSided(True)
    dome.setBin("background", 0)
    dome.setDepthWrite(False)
    dome.setDepthTest(False)
    dome.setShader(Shader.make(Shader.SL_GLSL, SKY_VS, SKY_FS))
    return dome


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------
def v3(a):
    return Vec3(float(a[0]), float(a[1]), float(a[2]))


def npv(v):
    return np.array([v[0], v[1], v[2]], dtype=np.float64)


class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        self.camLens.setNearFar(0.3, CAM_FAR)
        self.camLens.setFov(70)
        self.setBackgroundColor(0, 0, 0, 1)

        loading = OnscreenText(text="Generating planet...", scale=0.08, fg=(1, 1, 1, 1))
        for _ in range(2):
            self.graphicsEngine.renderFrame()
        loading.destroy()

        # --- choose the landing pad site and light the scene so it's morning there
        t0 = time.perf_counter()
        self.pad_dir = terrain.find_pad_site()
        pad_h = terrain.height_at(self.pad_dir) + 1.0
        terrain.set_pad(self.pad_dir, pad_h)
        self.pad_pos = self.pad_dir * (R + pad_h)
        east = np.cross([0.0, 0.0, 1.0], self.pad_dir)
        east /= np.linalg.norm(east)
        north = np.cross(self.pad_dir, east)
        sun = self.pad_dir * 0.55 + east * 0.75 + north * 0.35
        self.sun_dir = sun / np.linalg.norm(sun)

        render = self.render
        render.setShader(Shader.make(Shader.SL_GLSL, LIT_VS, LIT_FS))
        render.setShaderInput("sunDir", v3(self.sun_dir))
        render.setShaderInput("camPos", Vec3(0, 0, 0))
        render.setShaderInput("logFC", 2.0 / math.log2(CAM_FAR + 1.0))
        render.setShaderInput("emissive", 0.0)
        render.setShaderInput("tint", Vec4(1, 1, 1, 1))

        self.sky = build_sky_dome(render)
        self.planet = Planet(render)

        self.pad = build_pad(render)
        self.pad.setPos(v3(self.pad_pos))
        self.pad.lookAt(v3(self.pad_pos + north), v3(self.pad_dir))

        self.ship, self.flames, self.jets = build_ship(render)
        self.flash = None

        self.cam_mode = 0
        self.flight_assist = True
        self.throttle = 0.0
        self.in_vert = self.in_lat = 0.0
        self.boost = False
        self.mouse_ready = False
        self.show_help = True
        self.setup_hud()
        self.reset_ship()
        self.cam_pos = npv(self.ship.getPos(render)) - npv(self.ship.getQuat(render).getForward()) * 30
        self.cam_up = self.pad_dir.copy()
        self.ground_ship = self.ground_cam = 0.0
        self.update_camera(0.0, snap=True)

        # Build the LOD around the start position before the first frame
        while self.planet.update(self.camera.getPos(render), sync=True):
            pass
        print(f"Planet ready in {time.perf_counter() - t0:.1f}s, {self.planet.n_chunks} chunks")

        self.capture_mouse()
        self.accept("escape", sys.exit)
        self.accept("f", self.toggle_fa)
        self.accept("v", self.toggle_cam)
        self.accept("r", self.reset_ship)
        self.accept("x", self.cut_throttle)
        self.accept("h", self.toggle_help)
        self.taskMgr.add(self.update, "update")

    # -- setup ---------------------------------------------------------------
    def setup_hud(self):
        style = dict(fg=(0.75, 1.0, 0.8, 1), shadow=(0, 0, 0, 0.8), mayChange=True, scale=0.045)
        self.hud = OnscreenText(text="", pos=(-1.7, 0.92), align=0, parent=self.aspect2d, **style)
        self.hud.reparentTo(self.a2dTopLeft)
        self.hud.setPos(0.05, -0.08)
        self.status = OnscreenText(text="", pos=(0, 0.6), scale=0.08, fg=(1, 0.9, 0.4, 1),
                                   shadow=(0, 0, 0, 0.9), mayChange=True)
        self.help = OnscreenText(
            text="Mouse/Arrows: pitch & yaw   Q/E: roll   W/S: throttle   X: cut throttle\n"
                 "Space/C: thrust up/down   A/D: strafe   Shift: boost\n"
                 "F: flight assist   V: camera   R: back to pad   H: hide help   Esc: quit",
            pos=(0, -0.85), scale=0.045, fg=(1, 1, 1, 0.85), shadow=(0, 0, 0, 0.8), mayChange=True)
        # crosshair + pad direction marker
        self.crosshair = OnscreenText(text="+", pos=(0, -0.02), scale=0.06, fg=(0.7, 1, 0.8, 0.6))
        self.pad_marker = OnscreenText(text="[ PAD ]", scale=0.04, fg=(0.4, 0.8, 1, 0.9), mayChange=True,
                                       shadow=(0, 0, 0, 0.8))

    def capture_mouse(self):
        if SCREENSHOT_MODE:
            return
        props = WindowProperties()
        props.setCursorHidden(True)
        props.setMouseMode(WindowProperties.M_confined)
        self.win.requestProperties(props)
        self.mouse_ready = False

    def reset_ship(self):
        p = self.pad_pos + self.pad_dir * (-LEG_Z + 0.2)
        self.ship.setPos(v3(p))
        east = np.cross([0.0, 0.0, 1.0], self.pad_dir)
        self.ship.lookAt(v3(p + east), v3(self.pad_dir))
        self.ship.show()
        self.vel = np.zeros(3)
        self.throttle = 0.0
        self.landed = True
        self.crashed = 0.0
        self.message("", 0)
        self.thrust_local = np.zeros(3)

    def toggle_fa(self):
        self.flight_assist = not self.flight_assist
        self.message("FLIGHT ASSIST " + ("ON" if self.flight_assist else "OFF"), 1.5)

    def toggle_cam(self):
        self.cam_mode = (self.cam_mode + 1) % 3

    def cut_throttle(self):
        self.throttle = 0.0

    def toggle_help(self):
        self.show_help = not self.show_help
        if self.show_help:
            self.help.show()
        else:
            self.help.hide()

    def message(self, text, duration):
        self.status.setText(text)
        self.msg_timer = duration

    def is_down(self, key):
        if self.mouseWatcherNode is None:
            return False
        if isinstance(key, str):
            key = KeyboardButton.asciiKey(key)
        return self.mouseWatcherNode.isButtonDown(key)

    # -- per-frame -----------------------------------------------------------
    def update(self, task):
        dt = min(globalClock.getDt(), 0.05)  # noqa: F821  (ShowBase builtin)
        if self.crashed > 0:
            self.crashed -= dt
            self.update_flash(dt)
            if self.crashed <= 0:
                self.reset_ship()
        else:
            self.sample_ground()
            self.handle_input(dt)
            self.physics(dt)
        self.update_camera(dt)
        cam = self.camera.getPos(self.render)
        self.sky.setPos(cam)
        self.render.setShaderInput("camPos", cam)
        self.planet.update(cam)
        self.update_hud(dt)
        return task.cont

    def handle_input(self, dt):
        dh = dp = dr = 0.0
        if not SCREENSHOT_MODE and self.win.getProperties().getForeground():
            md = self.win.getPointer(0)
            cx, cy = self.win.getXSize() // 2, self.win.getYSize() // 2
            if md.getInWindow():
                if self.mouse_ready:
                    dh = -(md.getX() - cx) * MOUSE_SENS
                    dp = -(md.getY() - cy) * MOUSE_SENS
                self.mouse_ready = self.win.movePointer(0, cx, cy)
        dh = max(-YAW_RATE * 2 * dt, min(YAW_RATE * 2 * dt, dh))
        dp = max(-PITCH_RATE * 2 * dt, min(PITCH_RATE * 2 * dt, dp))
        kb = KeyboardButton
        if self.is_down(kb.up()):
            dp -= PITCH_RATE * dt
        if self.is_down(kb.down()):
            dp += PITCH_RATE * dt
        if self.is_down(kb.left()):
            dh += YAW_RATE * dt
        if self.is_down(kb.right()):
            dh -= YAW_RATE * dt
        if self.is_down("q"):
            dr -= ROLL_RATE * dt
        if self.is_down("e"):
            dr += ROLL_RATE * dt
        if self.landed:
            dp = dr = 0.0        # only yaw while sitting on the ground
        if dh or dp or dr:
            self.ship.setHpr(self.ship, dh, dp, dr)

        if self.is_down("w"):
            self.throttle = min(1.0, self.throttle + 0.6 * dt)
        if self.is_down("s"):
            self.throttle = max(-0.3 if self.flight_assist else 0.0, self.throttle - 0.6 * dt)
        self.in_vert = (1.0 if self.is_down(kb.space()) else 0.0) - (
            1.0 if (self.is_down("c") or self.is_down(kb.control())) else 0.0)
        self.in_lat = (1.0 if self.is_down("d") else 0.0) - (1.0 if self.is_down("a") else 0.0)
        self.boost = self.is_down(kb.shift())

    def sample_ground(self):
        """One batched terrain lookup per frame: ground under the ship and the camera."""
        sp = npv(self.ship.getPos(self.render))
        dirs = np.array([sp / np.linalg.norm(sp), self.cam_pos / np.linalg.norm(self.cam_pos)])
        h = np.maximum(terrain.height(dirs, 8.0), 0.0)
        self.ground_ship, self.ground_cam = float(h[0]), float(h[1])

    def physics(self, dt):
        render = self.render
        pos = npv(self.ship.getPos(render))
        quat = self.ship.getQuat(render)
        fwd, right, up = npv(quat.getForward()), npv(quat.getRight()), npv(quat.getUp())
        r = np.linalg.norm(pos)
        radial = pos / r
        g_vec = -radial * G0 * (R / r) ** 2
        ground_h = self.ground_ship
        self.agl = r - (R + ground_h) + LEG_Z
        self.alt = r - R

        if self.landed and self.throttle <= 0 and self.in_vert <= 0 and not self.in_lat:
            self.thrust_local = np.zeros(3)
            self.vel = np.zeros(3)
            return

        if self.flight_assist:
            fwd_speed = self.throttle * (FA_BOOST_SPEED if self.boost else FA_MAX_SPEED)
            vert_speed = self.in_vert * FA_VERT_SPEED
            if self.in_vert < 0:
                vert_speed = -min(FA_VERT_SPEED, 3.0 + max(self.agl, 0.0) * 0.4)   # gentle near ground
            target = fwd * fwd_speed + right * self.in_lat * FA_LAT_SPEED + up * vert_speed
        rho = math.exp(-max(self.alt, 0.0) / DRAG_H)
        n_sub = max(1, int(math.ceil(dt / (1 / 120))))
        h = dt / n_sub
        main_max = BOOST_ACC if self.boost else MAIN_ACC
        for _ in range(n_sub):
            speed = np.linalg.norm(self.vel)
            drag = -self.vel * speed * DRAG_K * rho
            if self.flight_assist:
                want = (target - self.vel) * 1.6 - g_vec - drag
                loc = np.array([want @ right, want @ fwd, want @ up])
                loc[0] = np.clip(loc[0], -LAT_ACC, LAT_ACC)
                loc[1] = np.clip(loc[1], -RETRO_ACC, main_max)
                loc[2] = np.clip(loc[2], -DOWN_ACC, VERT_ACC)
            else:
                loc = np.array([self.in_lat * LAT_ACC,
                                self.throttle * main_max,
                                self.in_vert * (VERT_ACC if self.in_vert > 0 else DOWN_ACC)])
            acc = right * loc[0] + fwd * loc[1] + up * loc[2] + g_vec + drag
            self.vel = self.vel + acc * h
            pos = pos + self.vel * h
        self.thrust_local = loc
        self.ship.setPos(v3(pos))
        self.landed = False
        self.collide(pos, quat, dt)

    def collide(self, pos, quat, dt):
        r = np.linalg.norm(pos)
        radial = pos / r
        if r - R - LEG_Z - 4.0 > 3000.0:
            return
        mat = self.ship.getMat(self.render)
        pts_local = [(3.0, -1.75, LEG_Z), (-3.0, -1.75, LEG_Z), (0.0, 3.75, LEG_Z),
                     (0.0, 6.5, 0.0), (0.0, -6.0, -0.8), (6.0, -1.5, 0.0), (-6.0, -1.5, 0.0),
                     (0.0, 0.0, 1.8)]
        pts = np.array([npv(mat.xformPoint(Point3(*p))) for p in pts_local])
        # cheap reject: lowest hull point way above the max possible terrain under us
        if r - 8.0 - (R + self.ground_ship) > 40.0 + np.linalg.norm(self.vel) * dt * 2:
            return
        # sample hull points + two extra points for the ground normal under the ship
        t1 = np.cross(radial, [0.0, 0.0, 1.0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(radial, t1)
        pr = np.linalg.norm(pts, axis=1, keepdims=True)
        dirs = np.concatenate([pts / pr, [radial, radial + t1 * 3.0 / R, radial + t2 * 3.0 / R]])
        dirs[-2:] /= np.linalg.norm(dirs[-2:], axis=1, keepdims=True)
        hs = terrain.height(dirs, 3.0)
        surf = R + np.maximum(hs[:len(pts)], 0.0)
        pen = surf - pr[:, 0]
        k = int(np.argmax(pen))
        if pen[k] <= 0:
            return
        # ground normal from three samples (water is flat)
        gp = [dirs[-3 + i] * (R + max(hs[-3 + i], 0.0)) for i in range(3)]
        gn = np.cross(gp[1] - gp[0], gp[2] - gp[0])
        gn /= np.linalg.norm(gn)
        if gn @ radial < 0:
            gn = -gn
        up = npv(quat.getUp())
        vn = self.vel @ radial
        vh = np.linalg.norm(self.vel - radial * vn)
        legs_down = k < 3 and up @ gn > math.cos(math.radians(30))
        on_water = hs[k] < 0
        speed = np.linalg.norm(self.vel)
        if (legs_down and -vn < 6.0 and vh < 8.0) or (not legs_down and speed < 4.0):
            pos = pos + radial * pen[k]
            if vn < 0:
                self.vel -= radial * vn
            self.vel -= (self.vel - radial * (self.vel @ radial)) * min(1.0, dt * 4.0)
            self.ship.setPos(v3(pos))
            if legs_down:
                # settle the ship flat on the ground
                cur_up = up
                axis = np.cross(cur_up, gn)
                s = np.linalg.norm(axis)
                if s > 1e-4:
                    ang = math.degrees(math.asin(min(1.0, s))) * min(1.0, dt * 5.0)
                    q = self.ship.getQuat(self.render)
                    rot = Quat()
                    rot.setFromAxisAngle(ang, v3(axis / s))
                    self.ship.setQuat(self.render, q * rot)
                idle = self.in_vert <= 0 and not self.in_lat and (
                    self.throttle <= 0.05 if self.flight_assist else self.thrust_local[2] < G0)
                if np.linalg.norm(self.vel) < 1.5 and idle:
                    self.vel = np.zeros(3)
                    self.throttle = min(self.throttle, 0.0) if self.flight_assist else 0.0
                    if not self.landed:
                        dist = np.linalg.norm(pos - self.pad_pos)
                        where = "ON THE PAD" if dist < 40 else ("ON WATER" if on_water else f"{dist / 1000:.1f} km FROM PAD")
                        self.message("LANDED " + where, 3.0)
                    self.landed = True
        else:
            self.crash(on_water)

    def crash(self, on_water):
        self.crashed = 3.0
        self.ship.hide()
        self.message("SPLASHED!" if on_water else "CRASHED!", 3.0)
        mb = MeshBuilder()
        for i in range(10):
            a = 2 * math.pi * i / 10
            for j in range(5):
                b0, b1 = math.pi * j / 5, math.pi * (j + 1) / 5
                a1 = a + 2 * math.pi / 10

                def p(aa, bb):
                    return (math.sin(bb) * math.cos(aa), math.sin(bb) * math.sin(aa), math.cos(bb))
                mb.quad(p(a, b0), p(a1, b0), p(a1, b1), p(a, b1), (1.0, 0.6, 0.2, 1))
        if self.flash:
            self.flash.removeNode()
        self.flash = self.render.attachNewNode(mb.node("flash"))
        self.flash.setTwoSided(True)
        self.flash.setPos(self.ship.getPos(self.render))
        self.flash.setShaderInput("emissive", 1.0)
        self.flash.setTransparency(TransparencyAttrib.MAlpha)
        self.flash.setDepthWrite(False)
        self.flash_t = 0.0
        self.vel = np.zeros(3)
        for f in self.flames + self.jets:
            f.hide()

    def update_flash(self, dt):
        if not self.flash:
            return
        self.flash_t += dt
        self.flash.setScale(3 + self.flash_t * 40)
        a = max(0.0, 1.0 - self.flash_t / 1.2)
        self.flash.setShaderInput("tint", Vec4(1, 1, 1, a))
        if a <= 0:
            self.flash.removeNode()
            self.flash = None

    def update_camera(self, dt, snap=False):
        render = self.render
        sp = npv(self.ship.getPos(render))
        q = self.ship.getQuat(render)
        fwd, up = npv(q.getForward()), npv(q.getUp())
        if self.cam_mode == 2:   # cockpit
            pos = sp + fwd * 2.2 + up * 1.6
            self.cam_pos, self.cam_up = pos, up
            self.camera.setPos(v3(pos))
            self.camera.lookAt(v3(pos + fwd * 100), v3(up))
            return
        dist, height = (28.0, 7.0) if self.cam_mode == 0 else (70.0, 18.0)
        want = sp - fwd * dist + up * height
        k = 1.0 if snap else 1.0 - math.exp(-dt * 6.0)
        # follow in the ship's moving frame so high speeds don't stretch the camera away
        self.cam_pos = self.cam_pos + self.vel * dt
        self.cam_pos = self.cam_pos + (want - self.cam_pos) * k
        self.cam_up = self.cam_up + (up - self.cam_up) * (1.0 if snap else 1.0 - math.exp(-dt * 4.0))
        self.cam_up /= np.linalg.norm(self.cam_up)
        # keep the camera above ground
        cr = np.linalg.norm(self.cam_pos)
        cdir = self.cam_pos / cr
        if cr - R < 4000:
            gh = self.ground_cam
            if cr < R + gh + 2.0:
                self.cam_pos = cdir * (R + gh + 2.0)
        self.camera.setPos(v3(self.cam_pos))
        self.camera.lookAt(v3(sp + fwd * 12.0), v3(self.cam_up))

    def update_hud(self, dt):
        render = self.render
        pos = npv(self.ship.getPos(render))
        r = np.linalg.norm(pos)
        radial = pos / r
        vn = self.vel @ radial
        vh = np.linalg.norm(self.vel - radial * vn)
        speed = np.linalg.norm(self.vel)
        alt = r - R
        agl = getattr(self, "agl", 0.0)
        if self.crashed > 0:
            state = "DESTROYED"
        elif self.landed:
            state = "LANDED"
        elif alt > SPACE_ALT:
            state = "IN SPACE"
        else:
            state = "FLYING"
        pad_d = np.linalg.norm(pos - self.pad_pos)
        orbit_v = math.sqrt(G0 * R * R / r)
        self.hud.setText(
            f"{state}\n"
            f"ALT  {alt:8.0f} m\n"
            f"AGL  {max(agl, 0):8.0f} m\n"
            f"VSPD {vn:8.1f} m/s\n"
            f"GSPD {vh:8.1f} m/s\n"
            f"SPD  {speed:8.1f} m/s  (orbit {orbit_v:.0f})\n"
            f"THR  {self.throttle * 100:6.0f} %{'  BOOST' if getattr(self, 'boost', False) else ''}\n"
            f"FA   {'ON' if self.flight_assist else 'OFF'}\n"
            f"PAD  {pad_d / 1000:8.2f} km\n"
            f"LOD  {self.planet.n_chunks} chunks"
        )
        if self.msg_timer > 0:
            self.msg_timer -= dt
            if self.msg_timer <= 0:
                self.status.setText("")
        elif not self.crashed and not self.landed and agl < 150 and vn < -25 and alt < SPACE_ALT:
            self.status.setText("PULL UP")
        elif self.status.getText() == "PULL UP":
            self.status.setText("")

        # project the pad position onto the screen
        p3 = self.camera.getRelativePoint(render, v3(self.pad_pos + self.pad_dir * 5))
        p2 = Point3()
        if p3.y > 0 and self.camLens.project(p3, p2) and pad_d > 120:
            self.pad_marker.show()
            self.pad_marker.setPos(p2.x * self.getAspectRatio(), p2.y)
            self.pad_marker.setText(f"[ PAD {pad_d / 1000:.1f} km ]")
        else:
            self.pad_marker.hide()

        # engine flames follow the thrust
        tl = self.thrust_local if self.crashed <= 0 else np.zeros(3)
        main = max(tl[1], 0.0) / BOOST_ACC
        flick = 0.85 + 0.3 * math.sin(globalClock.getFrameTime() * 60)  # noqa: F821
        for f in self.flames:
            if main > 0.02:
                f.show()
                f.setScale(0.6 + main, 1.0 + 9.0 * main * flick, 0.6 + main)
            else:
                f.hide()
        vert = max(tl[2], 0.0) / VERT_ACC
        for j in self.jets:
            if vert > 0.05:
                j.show()
                j.setScale(0.6 + vert * 0.6, 0.5 + 3.5 * vert * flick, 0.6 + vert * 0.6)
            else:
                j.hide()


def screenshot_test(game):
    """Headless smoke test: render a few views of the planet and save them."""
    out = sys.argv[sys.argv.index("--screenshot") + 1] if len(sys.argv) > sys.argv.index("--screenshot") + 1 else "shot"
    shots = []

    def place(alt, pitch, name, cam_mode=1):
        def run(task):
            if alt is not None:
                p = game.pad_pos + game.pad_dir * alt
                game.ship.setPos(v3(p))
                east = np.cross([0.0, 0.0, 1.0], game.pad_dir)
                game.ship.lookAt(v3(p + east), v3(game.pad_dir))
                game.ship.setP(game.ship, pitch)
                game.landed = alt < 5
                game.vel = np.zeros(3)
                game.cam_mode = cam_mode
                game.update_camera(0, snap=True)
                while game.planet.update(game.camera.getPos(game.render), sync=True):
                    pass
            return task.done

        def snap(task):
            game.win.saveScreenshot(f"{out}_{name}.png")
            shots.append(name)
            return task.done
        return run, snap

    plan = [(None, 0, "pad", 0), (600, -10, "low", 1), (4000, -20, "high", 1), (60000, -60, "space", 1)]
    t = 0.3
    for alt, pitch, name, cm in plan:
        run, snap = place(alt, pitch, name, cm)
        game.taskMgr.doMethodLater(t, run, f"run{name}")
        game.taskMgr.doMethodLater(t + 0.5, snap, f"snap{name}")
        t += 1.0
    game.taskMgr.doMethodLater(t + 0.2, lambda task: sys.exit(0), "quit")


if __name__ == "__main__":
    game = Game()
    if SCREENSHOT_MODE:
        game.flight_assist = True
        screenshot_test(game)
    game.run()
