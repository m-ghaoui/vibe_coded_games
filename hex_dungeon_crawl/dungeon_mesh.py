"""Procedural textures and level geometry for the hex dungeon.

All textures are generated from noise at startup (no asset files). The level
is built as a handful of large Geoms, one per material, so there are only a
few draw calls however big the dungeon is.
"""

import math
import random

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    InternalName,
    Material,
    NodePath,
    PNMImage,
    SamplerState,
    Texture,
    TextureStage,
    Vec3,
)

import hexgrid as hg

CELL_SIZE = 2.0        # hex circumradius: centre to corner
WALL_HEIGHT = 3.2
POST_RADIUS = 0.13     # posts where walls meet
PILLAR_RADIUS = 0.30   # free-standing hall pillars
TEX_SIZE = 256

# Stairs pit: terraces stepping down towards a glowing hole.
PIT_RADII = [1.55 - 0.22 * i for i in range(6)]
PIT_STEP = 0.3

_FORMAT = None


def _vertex_format():
    global _FORMAT
    if _FORMAT is None:
        arr = GeomVertexArrayFormat()
        arr.addColumn(InternalName.getVertex(), 3, Geom.NTFloat32, Geom.CPoint)
        arr.addColumn(InternalName.getNormal(), 3, Geom.NTFloat32, Geom.CNormal)
        arr.addColumn(InternalName.getTexcoord(), 2, Geom.NTFloat32, Geom.CTexcoord)
        arr.addColumn(InternalName.getTangent(), 3, Geom.NTFloat32, Geom.CVector)
        arr.addColumn(InternalName.getBinormal(), 3, Geom.NTFloat32, Geom.CVector)
        _FORMAT = GeomVertexFormat.registerFormat(arr)
    return _FORMAT


class MeshBuilder:
    """Collects triangles with normals, uvs and tangent frames."""

    def __init__(self, name):
        self.name = name
        self.vdata = GeomVertexData(name, _vertex_format(), Geom.UHStatic)
        self.vertex = GeomVertexWriter(self.vdata, "vertex")
        self.normal = GeomVertexWriter(self.vdata, "normal")
        self.uv = GeomVertexWriter(self.vdata, "texcoord")
        self.tangent = GeomVertexWriter(self.vdata, "tangent")
        self.binormal = GeomVertexWriter(self.vdata, "binormal")
        self.tris = GeomTriangles(Geom.UHStatic)
        self.count = 0

    def add_vertex(self, p, n, uv, t, b):
        self.vertex.addData3(*p)
        self.normal.addData3(*n)
        self.uv.addData2(*uv)
        self.tangent.addData3(*t)
        self.binormal.addData3(*b)
        self.count += 1
        return self.count - 1

    def quad(self, ps, n, uvs, t, b):
        """Four points in counter-clockwise order as seen from the front."""
        i = self.count
        for p, uv in zip(ps, uvs):
            self.add_vertex(p, n, uv, t, b)
        self.tris.addVertices(i, i + 1, i + 2)
        self.tris.addVertices(i, i + 2, i + 3)

    def node(self):
        geom = Geom(self.vdata)
        geom.addPrimitive(self.tris)
        gn = GeomNode(self.name)
        gn.addGeom(geom)
        return NodePath(gn)

    # -- shapes ---------------------------------------------------------

    def wall(self, left, right, z0, z1, uv_scale=0.5):
        """Vertical quad from left to right (as seen by the viewer it faces)."""
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        length = math.hypot(dx, dy)
        t = (dx / length, dy / length, 0.0)
        n = (t[1], -t[0], 0.0)   # facing the viewer
        u1 = length * uv_scale
        v0 = z0 * uv_scale
        v1 = z1 * uv_scale
        self.quad(
            [(left[0], left[1], z0), (right[0], right[1], z0),
             (right[0], right[1], z1), (left[0], left[1], z1)],
            n, [(0, v0), (u1, v0), (u1, v1), (0, v1)], t, (0, 0, 1))

    def hex_fan(self, cx, cy, r, z, up=True, uv_scale=0.25):
        """A flat hexagon (flat-top). Faces up, or down for ceilings."""
        pts = [(cx + r * math.cos(math.radians(60 * j)),
                cy + r * math.sin(math.radians(60 * j))) for j in range(6)]
        n = (0, 0, 1) if up else (0, 0, -1)
        c = self.add_vertex((cx, cy, z), n, (cx * uv_scale, cy * uv_scale), (1, 0, 0), (0, 1, 0))
        idx = [self.add_vertex((x, y, z), n, (x * uv_scale, y * uv_scale), (1, 0, 0), (0, 1, 0))
               for x, y in pts]
        for j in range(6):
            a, b = idx[j], idx[(j + 1) % 6]
            if up:
                self.tris.addVertices(c, a, b)
            else:
                self.tris.addVertices(c, b, a)

    def hex_ring(self, cx, cy, r_out, r_in, z, up=True, uv_scale=0.25):
        """A flat hexagonal annulus."""
        n = (0, 0, 1) if up else (0, 0, -1)
        for j in range(6):
            a0 = math.radians(60 * j)
            a1 = math.radians(60 * (j + 1))
            ps = [(cx + r_in * math.cos(a0), cy + r_in * math.sin(a0), z),
                  (cx + r_out * math.cos(a0), cy + r_out * math.sin(a0), z),
                  (cx + r_out * math.cos(a1), cy + r_out * math.sin(a1), z),
                  (cx + r_in * math.cos(a1), cy + r_in * math.sin(a1), z)]
            if not up:
                ps.reverse()
            self.quad(ps, n, [(p[0] * uv_scale, p[1] * uv_scale) for p in ps], (1, 0, 0), (0, 1, 0))

    def hex_tube(self, cx, cy, r, z0, z1, inward=False, rot=0.0):
        """Six vertical faces around a point: a prism (outward) or a pit (inward)."""
        for j in range(6):
            a0 = math.radians(rot + 60 * j)
            a1 = math.radians(rot + 60 * (j + 1))
            p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
            p1 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
            if inward:
                self.wall(p1, p0, z0, z1)
            else:
                self.wall(p0, p1, z0, z1)


# -- textures ---------------------------------------------------------------

def _fractal_noise(n, rng, octaves=((4, 0.5), (16, 0.3), (64, 0.2))):
    """Tileable value noise in [0, 1], as a list of rows."""
    out = [[0.0] * n for _ in range(n)]
    for cells, weight in octaves:
        grid = [[rng.random() for _ in range(cells)] for _ in range(cells)]
        step = n / cells
        for y in range(n):
            fy = y / step
            y0 = int(fy)
            ty = fy - y0
            ty = ty * ty * (3 - 2 * ty)
            row0 = grid[y0 % cells]
            row1 = grid[(y0 + 1) % cells]
            orow = out[y]
            for x in range(n):
                fx = x / step
                x0 = int(fx)
                tx = fx - x0
                tx = tx * tx * (3 - 2 * tx)
                x1 = (x0 + 1) % cells
                x0 %= cells
                top = row0[x0] + (row0[x1] - row0[x0]) * tx
                bot = row1[x0] + (row1[x1] - row1[x0]) * tx
                orow[x] += weight * (top + (bot - top) * ty)
    return out


def _to_texture(name, img):
    tex = Texture(name)
    tex.load(img)
    tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    tex.setMagfilter(SamplerState.FT_linear)
    tex.setAnisotropicDegree(4)
    return tex


def _stone_textures(name, rng, block_w, block_h, offset_rows, mortar, base, bump, bevel=4.0):
    """Blocks with mortar joints: returns (colour texture, normal map)."""
    n = TEX_SIZE
    noise = _fractal_noise(n, rng)
    height = [[0.0] * n for _ in range(n)]
    color = PNMImage(n, n, 3)
    shades = {}
    for y in range(n):
        row = y // block_h
        yy = y % block_h
        off = block_w // 2 if (offset_rows and row % 2) else 0
        nrow = noise[y]
        hrow = height[y]
        for x in range(n):
            xx = (x + off) % n
            bx = xx % block_w
            dm = min(yy, block_h - 1 - yy, bx, block_w - 1 - bx)
            nv = nrow[x]
            if dm < mortar:
                h = 0.15 * nv
                c = 0.22 + 0.15 * nv
            else:
                key = (row, xx // block_w)
                s = shades.get(key)
                if s is None:
                    s = shades[key] = 0.8 + 0.25 * rng.random()
                e = min(1.0, (dm - mortar + 1) / bevel)
                h = e * (0.7 + 0.3 * nv)
                c = s * (0.6 + 0.5 * nv) * (0.8 + 0.2 * e)
            hrow[x] = h
            color.setXel(x, y, min(1.0, base[0] * c), min(1.0, base[1] * c), min(1.0, base[2] * c))

    normal = PNMImage(n, n, 3)
    for y in range(n):
        up = height[y - 1]
        down = height[(y + 1) % n]
        row = height[y]
        for x in range(n):
            nx = -(row[(x + 1) % n] - row[x - 1]) * bump
            ny = (down[x] - up[x]) * bump      # image rows run top to bottom
            inv = 1.0 / math.sqrt(nx * nx + ny * ny + 1.0)
            normal.setXel(x, y, 0.5 + 0.5 * nx * inv, 0.5 + 0.5 * ny * inv, 0.5 + 0.5 * inv)
    return _to_texture(name, color), _to_texture(name + "-normal", normal)


class LevelTextures:
    """Generated once at startup and shared by every level."""

    def __init__(self, seed=7):
        rng = random.Random(seed)
        self.wall, self.wall_normal = _stone_textures(
            "wall", rng, 64, 32, True, 3, (0.56, 0.53, 0.50), bump=2.5)
        self.floor, self.floor_normal = _stone_textures(
            "floor", rng, 128, 64, True, 3, (0.55, 0.53, 0.50), bump=2.0, bevel=6.0)
        self.ceiling, _ = _stone_textures(
            "ceiling", rng, 128, 128, False, 4, (0.42, 0.40, 0.38), bump=1.0, bevel=10.0)
        self.normal_stage = TextureStage("normal")
        self.normal_stage.setMode(TextureStage.MNormal)


# -- level --------------------------------------------------------------------

class Level:
    def __init__(self, root, beacon, stairs_pos):
        self.root = root
        self.beacon = beacon
        self.stairs_pos = stairs_pos


def _material(specular, shininess):
    m = Material()
    m.setDiffuse((1, 1, 1, 1))
    m.setAmbient((1, 1, 1, 1))
    m.setSpecular((specular, specular, specular, 1))
    m.setShininess(shininess)
    return m


def build_level(dungeon, textures, tint=(1, 1, 1, 1), glow_color=(0.45, 0.85, 1.0, 1)):
    walls = MeshBuilder("walls")
    floor = MeshBuilder("floor")
    ceiling = MeshBuilder("ceiling")
    glow = MeshBuilder("glow")
    size = CELL_SIZE

    for cell in sorted(dungeon.cells):
        cx, cy = hg.hex_to_world(cell, size)
        corners = [(cx + ox, cy + oy) for ox, oy in (hg.corner_offset(k, size) for k in range(6))]

        if cell == dungeon.stairs:
            r, z = PIT_RADII, [-PIT_STEP * i for i in range(len(PIT_RADII) + 1)]
            floor.hex_ring(cx, cy, size, r[0], 0.0)
            for i in range(len(r)):
                floor.hex_tube(cx, cy, r[i], z[i + 1], z[i], inward=True)
                if i + 1 < len(r):
                    floor.hex_ring(cx, cy, r[i], r[i + 1], z[i + 1])
            glow.hex_fan(cx, cy, r[-1], z[-1])
        else:
            floor.hex_fan(cx, cy, size, 0.0)
        ceiling.hex_fan(cx, cy, size, WALL_HEIGHT, up=False)

        for d in range(6):
            if not dungeon.is_open(cell, d):
                left, right = hg.edge_corners(d)
                walls.wall(corners[left], corners[right], 0.0, WALL_HEIGHT)

    # Posts where walls meet, and free-standing pillars in the halls.
    done = set()
    for cell in sorted(dungeon.cells):
        cx, cy = hg.hex_to_world(cell, size)
        for k in range(6):
            key = hg.vertex_key(cell, k)
            if key in done:
                continue
            done.add(key)
            ox, oy = hg.corner_offset(k, size)
            px, py = cx + ox, cy + oy
            if dungeon.vertex_wall_count(cell, k):
                walls.hex_tube(px, py, POST_RADIUS, 0.0, WALL_HEIGHT, rot=30)
            elif key in dungeon.pillars:
                walls.hex_tube(px, py, PILLAR_RADIUS, 0.0, WALL_HEIGHT, rot=30)
                for z0, z1 in ((0.0, 0.3), (WALL_HEIGHT - 0.3, WALL_HEIGHT)):
                    walls.hex_tube(px, py, PILLAR_RADIUS + 0.12, z0, z1, rot=30)
                    walls.hex_ring(px, py, PILLAR_RADIUS + 0.12, PILLAR_RADIUS, z1 if z0 == 0 else z0,
                                   up=(z0 == 0))

    root = NodePath("level")
    stone = root.attachNewNode("stone")
    stone.setColorScale(tint)
    for builder, tex, nmap, mat in (
            (walls, textures.wall, textures.wall_normal, _material(0.12, 24)),
            (floor, textures.floor, textures.floor_normal, _material(0.25, 40)),
            (ceiling, textures.ceiling, None, _material(0.0, 1))):
        np = builder.node()
        np.reparentTo(stone)
        np.setTexture(tex)
        if nmap is not None:
            np.setTexture(textures.normal_stage, nmap)
        np.setMaterial(mat)

    sx, sy = hg.hex_to_world(dungeon.stairs, size)
    glow_np = glow.node()
    glow_np.reparentTo(root)
    glow_np.setLightOff(1)
    glow_np.setColor(glow_color)
    glow_np.setTextureOff(1)

    # A slowly turning hexagonal halo floating above the stairs.
    halo = MeshBuilder("halo")
    halo.hex_ring(0, 0, 0.55, 0.45, 0.0, up=True)
    halo.hex_ring(0, 0, 0.55, 0.45, 0.0, up=False)
    beacon = halo.node()
    beacon.reparentTo(root)
    beacon.setPos(sx, sy, WALL_HEIGHT - 0.6)
    beacon.setLightOff(1)
    beacon.setFogOff(1)
    beacon.setTextureOff(1)
    beacon.setColor(glow_color)

    return Level(root, beacon, (sx, sy))
