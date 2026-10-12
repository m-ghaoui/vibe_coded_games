"""Procedural mesh helpers: flat-shaded, vertex-coloured triangles."""

import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    Point3,
    Vec3,
)


class MeshBuilder:
    """Accumulates flat-shaded, vertex-coloured triangles into one GeomNode."""

    def __init__(self, name="mesh"):
        self.name = name
        self.vdata = GeomVertexData(name, GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
        self.vertex = GeomVertexWriter(self.vdata, "vertex")
        self.normal = GeomVertexWriter(self.vdata, "normal")
        self.color = GeomVertexWriter(self.vdata, "color")
        self.tris = GeomTriangles(Geom.UHStatic)
        self.count = 0

    def tri(self, a, b, c, color, outward=None, normal=None):
        a, b, c = Point3(*a), Point3(*b), Point3(*c)
        face = (b - a).cross(c - a)
        if face.lengthSquared() < 1e-12:
            return
        if outward is not None and face.dot(Vec3(*outward)) < 0:
            b, c = c, b
            face = -face
        n = Vec3(*normal) if normal is not None else face.normalized()
        for p in (a, b, c):
            self.vertex.addData3(p)
            self.normal.addData3(n)
            self.color.addData4(*color)
        self.tris.addVertices(self.count, self.count + 1, self.count + 2)
        self.count += 3

    def quad(self, a, b, c, d, color, outward=None, normal=None):
        if outward is None:
            outward = (Point3(*b) - Point3(*a)).cross(Point3(*c) - Point3(*a))
        self.tri(a, b, c, color, outward, normal)
        self.tri(a, c, d, color, outward, normal)

    def node(self):
        geom = Geom(self.vdata)
        geom.addPrimitive(self.tris)
        gn = GeomNode(self.name)
        gn.addGeom(geom)
        return NodePath(gn)


def add_box(b, x0, x1, y0, y1, z0, z1, color):
    faces = [
        ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
        ((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (0, 1, 0)),
        ((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (-1, 0, 0)),
        ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (1, 0, 0)),
        ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1)),
        ((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0), (0, 0, -1)),
    ]
    for a, bb, c, d, out in faces:
        b.quad(a, bb, c, d, color, out)


def add_frustum(b, r0, r1, z0, z1, segs, color, cx=0.0, cy=0.0, jitter=None):
    """Cylinder/cone along Z. r1=0 gives a cone. jitter scales each ring point."""
    jitter = jitter or [1.0] * segs
    lo, hi = [], []
    for i in range(segs):
        ang = 2 * math.pi * i / segs
        ca, sa = math.cos(ang), math.sin(ang)
        lo.append((cx + ca * r0 * jitter[i], cy + sa * r0 * jitter[i], z0))
        hi.append((cx + ca * r1 * jitter[i], cy + sa * r1 * jitter[i], z1))
    for i in range(segs):
        j = (i + 1) % segs
        mid = 2 * math.pi * (i + 0.5) / segs
        out = (math.cos(mid), math.sin(mid), 0.0)
        if r1 == 0:
            b.tri(lo[i], lo[j], (cx, cy, z1), color, out)
        else:
            b.quad(lo[i], lo[j], hi[j], hi[i], color, out)
    for i in range(1, segs - 1):
        b.tri(lo[0], lo[i], lo[i + 1], color, (0, 0, -1))
        if r1 != 0:
            b.tri(hi[0], hi[i], hi[i + 1], color, (0, 0, 1))


def box_node(name, x0, x1, y0, y1, z0, z1, color):
    b = MeshBuilder(name)
    add_box(b, x0, x1, y0, y1, z0, z1, color)
    return b.node()


def shade(color, f):
    return (min(color[0] * f, 1), min(color[1] * f, 1), min(color[2] * f, 1), 1)
