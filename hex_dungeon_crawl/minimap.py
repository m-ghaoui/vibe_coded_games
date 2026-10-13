"""Fog-of-war hex map, drawn by its own orthographic camera into a corner
display region. M switches between the corner minimap and a big overlay."""

import math

from panda3d.core import (
    Camera,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    NodePath,
    OrthographicLens,
)

import hexgrid as hg
from dungeon_mesh import CELL_SIZE

SMALL_REGION = (0.785, 0.985, 0.62, 0.975)
BIG_REGION = (0.10, 0.90, 0.07, 0.93)
SMALL_VIEW = 30.0           # world units visible vertically in the minimap

SEEN_COLOR = (0.16, 0.15, 0.14, 1)
VISITED_COLOR = (0.30, 0.27, 0.23, 1)
WALL_COLOR = (0.85, 0.78, 0.62, 1)
STAIRS_COLOR = (0.45, 0.85, 1.0, 1)
PLAYER_COLOR = (1.0, 0.45, 0.2, 1)


def _hex_points(cx, cy, r):
    return [(cx + r * math.cos(math.radians(60 * j)), cy + r * math.sin(math.radians(60 * j)))
            for j in range(6)]


class Minimap:
    def __init__(self, base):
        self.base = base
        self.big = False
        self.dungeon = None
        self.scene = NodePath("map-scene")
        self.scene.setDepthTest(False)
        self.scene.setDepthWrite(False)
        self.scene.setLightOff(1)

        self.lens = OrthographicLens()
        self.lens.setNearFar(1, 100)
        self.cam = self.scene.attachNewNode(Camera("map-cam", self.lens))
        self.cam.setP(-90)

        self.region = base.win.makeDisplayRegion(*SMALL_REGION)
        self.region.setSort(5)       # after the 3D view, before the 2D HUD
        self.region.setCamera(self.cam)
        self.region.setClearColorActive(True)
        self.region.setClearColor((0.03, 0.03, 0.035, 1))
        self.region.setClearDepthActive(True)

        self.content = self.scene.attachNewNode("content")
        self.player = self._make_player()
        self.player.reparentTo(self.scene)
        self.player.setBin("fixed", 10)

    def _make_player(self):
        s = CELL_SIZE * 0.55
        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("player", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        for x, y in ((0, s), (-0.6 * s, -0.6 * s), (0.6 * s, -0.6 * s)):
            vw.addData3(x, y, 0)
            cw.addData4(*PLAYER_COLOR)
        tris = GeomTriangles(Geom.UHStatic)
        tris.addVertices(0, 1, 2)
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        gn = GeomNode("player")
        gn.addGeom(geom)
        np = NodePath(gn)
        np.setTwoSided(True)
        return np

    def set_level(self, dungeon):
        self.dungeon = dungeon

    def toggle(self):
        self.big = not self.big

    def rebuild(self, discovered, visited):
        """Redraw every discovered cell. Cheap enough to do once per step."""
        self.content.removeNode()
        self.content = self.scene.attachNewNode("content")
        dun = self.dungeon

        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData("map-floor", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        tris = GeomTriangles(Geom.UHStatic)
        count = 0
        walls = LineSegs("map-walls")
        walls.setThickness(2.0 if not self.big else 2.5)
        walls.setColor(WALL_COLOR)

        for cell in discovered:
            cx, cy = hg.hex_to_world(cell, CELL_SIZE)
            if cell == dun.stairs:
                color = STAIRS_COLOR
            elif cell in visited:
                color = VISITED_COLOR
            else:
                color = SEEN_COLOR
            vw.addData3(cx, cy, 0)
            cw.addData4(*color)
            for x, y in _hex_points(cx, cy, CELL_SIZE):
                vw.addData3(x, y, 0)
                cw.addData4(*color)
            for j in range(6):
                tris.addVertices(count, count + 1 + j, count + 1 + (j + 1) % 6)
            count += 7

            corners = [(cx + ox, cy + oy) for ox, oy in (hg.corner_offset(k, CELL_SIZE) for k in range(6))]
            for d in range(6):
                if not dun.is_open(cell, d):
                    left, right = hg.edge_corners(d)
                    walls.moveTo(corners[left][0], corners[left][1], 0)
                    walls.drawTo(corners[right][0], corners[right][1], 0)

        if count:
            geom = Geom(vdata)
            geom.addPrimitive(tris)
            gn = GeomNode("map-floor")
            gn.addGeom(geom)
            floor = self.content.attachNewNode(gn)
            floor.setTwoSided(True)
            floor.setBin("fixed", 0)
        wall_np = self.content.attachNewNode(walls.create())
        wall_np.setBin("fixed", 5)

    def update(self, player_pos, view_heading_deg):
        """Follow the player (small map) or show the whole level (big map)."""
        dims = BIG_REGION if self.big else SMALL_REGION
        self.region.setDimensions(*dims)
        win = self.base.win
        px_w = (dims[1] - dims[0]) * win.getXSize()
        px_h = (dims[3] - dims[2]) * win.getYSize()
        aspect = px_w / max(1.0, px_h)

        x, y = player_pos
        self.player.setPos(x, y, 0)
        self.player.setH(-view_heading_deg)
        self.player.setScale(1.0 if not self.big else 1.3)

        if self.big and self.dungeon is not None:
            r = self.dungeon.radius
            ext_x = 2 * (1.5 * CELL_SIZE * r + CELL_SIZE) + 4
            ext_y = 2 * (hg.SQRT3 * CELL_SIZE * r + CELL_SIZE) + 4
            view_h = max(ext_y, ext_x / aspect)
            self.cam.setPos(0, 0, 50)
        else:
            view_h = SMALL_VIEW
            self.cam.setPos(x, y, 50)
        self.lens.setFilmSize(view_h * aspect, view_h)
