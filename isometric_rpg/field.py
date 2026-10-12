"""The field: tiled ground, grass, rocks and trees, plus collision against them."""

import math

from panda3d.core import NodePath

from geometry import MeshBuilder, add_box, add_frustum, shade

FIELD_SIZE = 20

GRASS = (0.34, 0.56, 0.22, 1)
DIRT = (0.47, 0.36, 0.22, 1)
DIRT_DARK = (0.32, 0.23, 0.14, 1)
ROCK = (0.52, 0.52, 0.50, 1)
BARK = (0.38, 0.25, 0.14, 1)
PINE = (0.16, 0.38, 0.20, 1)


def build_field(rng):
    root = NodePath("field")
    half = FIELD_SIZE / 2

    ground = MeshBuilder("ground")
    for ix in range(FIELD_SIZE):
        for iy in range(FIELD_SIZE):
            x0, y0 = ix - half, iy - half
            cx, cy = x0 + 0.5, y0 + 0.5
            # Worn patch of dirt around where the warrior stands.
            wear = max(0.0, 1.0 - math.hypot(cx, cy) / 2.2) + rng.uniform(-0.25, 0.25)
            if wear > 0.45:
                base = DIRT
            else:
                base = GRASS
            f = rng.uniform(0.88, 1.08)
            col = (base[0] * f * rng.uniform(0.96, 1.04), base[1] * f, base[2] * f, 1)
            ground.quad((x0, y0, 0), (x0 + 1, y0, 0), (x0 + 1, y0 + 1, 0), (x0, y0 + 1, 0),
                        col, (0, 0, 1))
    # Diorama sides: earth with a darker layer underneath.
    for z0, z1, col in ((-0.35, 0, DIRT), (-0.9, -0.35, DIRT_DARK)):
        for a, b2, out in (((-half, -half), (half, -half), (0, -1, 0)),
                           ((half, -half), (half, half), (1, 0, 0)),
                           ((half, half), (-half, half), (0, 1, 0)),
                           ((-half, half), (-half, -half), (-1, 0, 0))):
            ground.quad((a[0], a[1], z0), (b2[0], b2[1], z0), (b2[0], b2[1], z1),
                        (a[0], a[1], z1), col, out)
    ground.node().reparentTo(root)

    def free_spot(margin, keep_clear):
        while True:
            x = rng.uniform(-half + margin, half - margin)
            y = rng.uniform(-half + margin, half - margin)
            if math.hypot(x, y) > keep_clear:
                return x, y

    # Grass tufts: thin two-sided blades with upward normals so they light like the ground.
    tufts = MeshBuilder("tufts")
    for _ in range(650):
        x, y = free_spot(0.2, 1.6)
        col = shade(GRASS, rng.uniform(1.0, 1.35))
        for _ in range(3):
            ang = rng.uniform(0, math.pi)
            dx, dy = math.cos(ang) * 0.035, math.sin(ang) * 0.035
            h = rng.uniform(0.12, 0.26)
            lx, ly = rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06)
            tufts.tri((x - dx, y - dy, 0), (x + dx, y + dy, 0), (x + lx, y + ly, h),
                      col, normal=(0, 0, 1))
    # A few flowers.
    for _ in range(40):
        x, y = free_spot(0.3, 2.0)
        col = rng.choice([(0.95, 0.85, 0.25, 1), (0.95, 0.95, 0.95, 1), (0.75, 0.45, 0.85, 1)])
        add_box(tufts, x - 0.03, x + 0.03, y - 0.03, y + 0.03, 0.12, 0.17, col)
        tufts.tri((x - 0.008, y, 0), (x + 0.008, y, 0), (x, y, 0.13), GRASS, normal=(0, 0, 1))
    tuft_np = tufts.node()
    tuft_np.setTwoSided(True)
    tuft_np.reparentTo(root)

    # Obstacles the warrior collides with, as (x, y, radius).
    obstacles = []
    props = MeshBuilder("props")
    for _ in range(14):
        x, y = free_spot(0.6, 2.5)
        s = rng.uniform(0.15, 0.4)
        segs = rng.randint(5, 7)
        jit = [rng.uniform(0.75, 1.2) for _ in range(segs)]
        add_frustum(props, s, s * 0.6, 0, s * rng.uniform(0.5, 0.9), segs,
                    shade(ROCK, rng.uniform(0.85, 1.1)), x, y, jit)
        obstacles.append((x, y, s * 0.9))
    for _ in range(7):
        x, y = free_spot(1.2, 4.0)
        s = rng.uniform(0.8, 1.25)
        add_frustum(props, 0.13 * s, 0.09 * s, 0, 0.7 * s, 6, BARK, x, y)
        obstacles.append((x, y, 0.4 * s))
        col = shade(PINE, rng.uniform(0.85, 1.15))
        for k, (r, z) in enumerate(((0.95, 0.5), (0.75, 1.1), (0.52, 1.65))):
            add_frustum(props, r * s, 0, z * s, (z + 0.95) * s, 8, shade(col, 1 + 0.08 * k), x, y)
    props.node().reparentTo(root)
    return root, obstacles


def resolve_collision(pos, radius, obstacles):
    """Keep a circle of `radius` at `pos` inside the field and out of obstacles."""
    limit = FIELD_SIZE / 2 - radius
    pos.x = max(-limit, min(limit, pos.x))
    pos.y = max(-limit, min(limit, pos.y))
    for ox, oy, r in obstacles:
        dx, dy = pos.x - ox, pos.y - oy
        dist = math.hypot(dx, dy)
        min_dist = r + radius
        if 1e-6 < dist < min_dist:
            pos.x = ox + dx / dist * min_dist
            pos.y = oy + dy / dist * min_dist
    return pos


def is_free(x, y, radius, obstacles):
    return all(math.hypot(x - ox, y - oy) > r + radius for ox, oy, r in obstacles)
