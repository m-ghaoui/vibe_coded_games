"""Hex grid maths and dungeon generation. Pure Python, no Panda3D imports.

The grid uses flat-top hexagons in axial coordinates (q, r), with +Y as north.
There are six headings, numbered clockwise:

    0 = N, 1 = NE, 2 = SE, 3 = S, 4 = SW, 5 = NW

Every cell has six edges, one per heading. Edge d of a cell is shared with
neighbor(cell, d). It is either open (a passage) or closed (a wall).
Corner k of a cell is the vertex between edge k and edge k+1. Three cells
meet at it: the cell itself, neighbor(cell, k) and neighbor(cell, k+1).
"""

import math
import random
from collections import deque

DIRS = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]
DIR_NAMES = ["N", "NE", "SE", "S", "SW", "NW"]
SQRT3 = math.sqrt(3.0)


def neighbor(cell, d):
    dq, dr = DIRS[d % 6]
    return (cell[0] + dq, cell[1] + dr)


def opposite(d):
    return (d + 3) % 6


def hex_to_world(cell, size):
    """Centre of a cell in world units (x east, y north)."""
    q, r = cell
    return (1.5 * size * q, SQRT3 * size * (r + q / 2.0))


def heading_vector(d):
    """Unit (x, y) vector pointing towards heading d."""
    a = math.radians(60.0 * d)
    return (math.sin(a), math.cos(a))


def corner_offset(k, size):
    """Offset from a cell centre to its corner k (between edges k and k+1)."""
    a = math.radians(60.0 - 60.0 * k)
    return (size * math.cos(a), size * math.sin(a))


def edge_corners(d):
    """Corner indices (left, right) of edge d, as seen from inside the cell."""
    return ((d - 1) % 6, d % 6)


def vertex_key(cell, k):
    """A key that is the same for all three cells that share a corner."""
    return frozenset((cell, neighbor(cell, k), neighbor(cell, k + 1)))


def cells_in_radius(radius, centre=(0, 0)):
    cq, cr = centre
    out = []
    for q in range(-radius, radius + 1):
        for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
            out.append((cq + q, cr + r))
    return out


def hex_distance(a, b):
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


class Dungeon:
    def __init__(self, radius, cells):
        self.radius = radius
        self.cells = set(cells)
        self.open = {c: [False] * 6 for c in self.cells}
        self.halls = []          # list of sets of cells
        self.pillars = set()     # vertex keys that carry a free-standing pillar
        self.start = None
        self.stairs = None

    def is_open(self, cell, d):
        edges = self.open.get(cell)
        return bool(edges and edges[d % 6])

    def set_open(self, cell, d, value=True):
        n = neighbor(cell, d)
        if cell in self.cells and n in self.cells:
            self.open[cell][d % 6] = value
            self.open[n][opposite(d)] = value

    def is_wall(self, a, b, d):
        """Is there a wall between a and its neighbour b = neighbor(a, d)?"""
        if a in self.cells:
            return not self.is_open(a, d)
        return b in self.cells

    def vertex_wall_count(self, cell, k):
        """Number of walls that meet at corner k of cell."""
        a = neighbor(cell, k)
        b = neighbor(cell, k + 1)
        return (self.is_wall(cell, a, k)
                + self.is_wall(cell, b, k + 1)
                + self.is_wall(a, b, k + 2))

    def hall_of(self, cell):
        for hall in self.halls:
            if cell in hall:
                return hall
        return None

    def distances(self, src):
        dist = {src: 0}
        todo = deque([src])
        while todo:
            c = todo.popleft()
            for d in range(6):
                if self.is_open(c, d):
                    n = neighbor(c, d)
                    if n not in dist:
                        dist[n] = dist[c] + 1
                        todo.append(n)
        return dist

    def ray(self, cell, d):
        """Cells seen looking straight along heading d (not including cell)."""
        out = []
        while self.is_open(cell, d):
            cell = neighbor(cell, d)
            out.append(cell)
        return out


def generate(radius, seed, depth=1):
    rng = random.Random(seed)
    dun = Dungeon(radius, cells_in_radius(radius))
    all_cells = sorted(dun.cells)

    # 1. A perfect maze (iterative recursive backtracker).
    first = rng.choice(all_cells)
    stack = [first]
    seen = {first}
    while stack:
        c = stack[-1]
        options = [d for d in range(6)
                   if neighbor(c, d) in dun.cells and neighbor(c, d) not in seen]
        if not options:
            stack.pop()
            continue
        d = rng.choice(options)
        n = neighbor(c, d)
        dun.set_open(c, d)
        seen.add(n)
        stack.append(n)

    # 2. Knock out some walls so the maze has loops.
    closed = [(c, d) for c in all_cells for d in range(3)
              if neighbor(c, d) in dun.cells and not dun.is_open(c, d)]
    rng.shuffle(closed)
    for c, d in closed[:int(len(closed) * (0.08 + 0.01 * min(depth, 5)))]:
        dun.set_open(c, d)

    # 3. Pillared halls: open every wall inside a small hex patch.
    centres = []
    wanted = 2 + radius // 4
    for _ in range(60):
        if len(centres) >= wanted:
            break
        hr = rng.choice([1, 1, 2]) if radius >= 6 else 1
        c = rng.choice(all_cells)
        if hex_distance(c, (0, 0)) > radius - hr:
            continue
        if any(hex_distance(c, oc) <= hr + ohr + 1 for oc, ohr in centres):
            continue
        centres.append((c, hr))
        hall = set(cells_in_radius(hr, c))
        for h in hall:
            for d in range(6):
                if neighbor(h, d) in hall:
                    dun.set_open(h, d)
        dun.halls.append(hall)
        for h in sorted(hall):
            for k in range(6):
                if neighbor(h, k) in hall and neighbor(h, k + 1) in hall:
                    if rng.random() < 0.75:
                        dun.pillars.add(vertex_key(h, k))

    # 4. Start in a corridor; the stairs go as far away as possible.
    corridor = [c for c in all_cells if dun.hall_of(c) is None]
    dun.start = rng.choice(corridor or all_cells)
    dist = dun.distances(dun.start)
    far = max(dist.values())
    dun.stairs = min(c for c, v in dist.items() if v == far)
    return dun


def visible_cells(dun, cell):
    """Cells the player can see from cell: straight lines in all six
    directions, plus the whole hall when standing in one."""
    seen = {cell}
    for d in range(6):
        seen.update(dun.ray(cell, d))
    hall = dun.hall_of(cell)
    if hall:
        seen |= hall
    return seen
