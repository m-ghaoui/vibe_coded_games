"""The player character: a procedurally built warrior with cloak and sword."""

import math

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    Vec3,
)

from geometry import MeshBuilder, add_box, box_node, shade

SKIN = (0.86, 0.66, 0.52, 1)
STEEL = (0.68, 0.70, 0.74, 1)
STEEL_DARK = (0.42, 0.44, 0.48, 1)
CHAIN = (0.50, 0.52, 0.56, 1)
LEATHER = (0.36, 0.22, 0.12, 1)
LEATHER_DARK = (0.22, 0.13, 0.07, 1)
CLOTH = (0.20, 0.28, 0.48, 1)
CLOAK = (0.60, 0.08, 0.10, 1)
GOLD = (0.88, 0.70, 0.25, 1)
BLADE = (0.86, 0.88, 0.92, 1)

ARM_R_PITCH = -12.0
RUN_SPEED = 4.5      # units per second
TURN_RATE = 720.0    # degrees per second
WARRIOR_RADIUS = 0.3
ELBOW_R_PITCH = -38.0
ATTACK_TIME = 0.4    # seconds per sword swing
ATTACK_HIT_AT = 0.5  # swing progress at which the blade connects

# Sword swing keyframes: (progress, arm_r P, arm_r R, elbow P, sword P, torso H, torso P).
# A diagonal chop: raise the sword high on the right, slash down across the body.
SWING_KEYS = [
    (0.00, -60, -15, -30, 175, 0, 3),
    (0.35, -165, -35, -25, 175, 28, -4),
    (0.50, -95, -5, -10, 180, 0, 8),
    (0.62, -35, 25, -5, 180, -30, 14),
    (1.00, -35, 25, -5, 180, -30, 14),
]


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def sample_keys(keys, p):
    """Smoothly interpolate a keyframe table; returns the values after the time column."""
    for k0, k1 in zip(keys, keys[1:]):
        if p <= k1[0]:
            f = smoothstep((p - k0[0]) / (k1[0] - k0[0]))
            return [a + (b - a) * f for a, b in zip(k0[1:], k1[1:])]
    return list(keys[-1][1:])


def lerp(a, b, f):
    return a + (b - a) * f


class Cloak:
    """A cloth grid hanging from the shoulders, rewritten every frame to ripple."""

    COLS, ROWS = 9, 13

    def __init__(self, parent):
        self.base = []
        for r in range(self.ROWS):
            v = r / (self.ROWS - 1)
            for c in range(self.COLS):
                s = 2 * c / (self.COLS - 1) - 1
                x = s * (0.25 + 0.15 * v)
                y = 0.15 + 0.14 * v ** 1.2 + 0.05 * (1 - s * s) - 0.05 * s * s * (1 - v)
                z = 0.47 - 1.32 * v
                self.base.append(Vec3(x, y, z))

        fmt = GeomVertexFormat.getV3n3c4()
        self.vdata = GeomVertexData("cloak", fmt, Geom.UHDynamic)
        self.vdata.setNumRows(len(self.base))
        col = GeomVertexWriter(self.vdata, "color")
        for r in range(self.ROWS):
            # Slightly darker toward the hem.
            f = 1.0 - 0.25 * r / (self.ROWS - 1)
            for _ in range(self.COLS):
                col.setData4(*shade(CLOAK, f))
        tris = GeomTriangles(Geom.UHStatic)
        for r in range(self.ROWS - 1):
            for c in range(self.COLS - 1):
                i = r * self.COLS + c
                # Wound so the outside (+Y, the back) is the front face.
                tris.addVertices(i, i + 1, i + self.COLS + 1)
                tris.addVertices(i, i + self.COLS + 1, i + self.COLS)
        self.geom = Geom(self.vdata)
        self.geom.addPrimitive(tris)
        gn = GeomNode("cloak")
        gn.addGeom(self.geom)
        self.np = parent.attachNewNode(gn)
        self.np.setTwoSided(True)
        self.update(0.0)

    def update(self, t, run=0.0):
        """run (0..1) streams the cloak out behind and makes it flutter faster."""
        pts = []
        flutter = 1.0 + 1.5 * run
        for r in range(self.ROWS):
            v = r / (self.ROWS - 1)
            amp = v ** 1.5
            # Swing the cloth back around the shoulder line, more toward the hem.
            ang = run * 0.65 * v ** 0.8
            for c in range(self.COLS):
                u = c / (self.COLS - 1)
                p = self.base[r * self.COLS + c]
                d = amp * (0.045 * math.sin(1.3 * flutter * t + 3.0 * u + 2.2 * v)
                           + 0.025 * math.sin(2.1 * flutter * t - 4.0 * v + 1.7 * u)) * flutter
                y, z = p.y - 0.15, p.z - 0.47
                y, z = y * math.cos(ang) - z * math.sin(ang), y * math.sin(ang) + z * math.cos(ang)
                pts.append(Vec3(p.x + 0.35 * d, y + 0.15 + d + 0.02 * amp,
                                z + 0.47 + 0.15 * abs(d)))

        vdata = self.geom.modifyVertexData()
        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        C, R = self.COLS, self.ROWS
        for r in range(R):
            for c in range(C):
                p = pts[r * C + c]
                du = pts[r * C + min(c + 1, C - 1)] - pts[r * C + max(c - 1, 0)]
                dv = pts[min(r + 1, R - 1) * C + c] - pts[max(r - 1, 0) * C + c]
                n = dv.cross(du)
                if n.y < 0:
                    n = -n
                n.normalize()
                vw.setData3(p)
                nw.setData3(n)


class Warrior:
    def __init__(self, parent):
        self.root = parent.attachNewNode("warrior")
        self.run = 0.0    # 0 = idle, 1 = full run (smoothed)
        self.phase = 0.0  # run cycle phase
        # Everything visible hangs off body, so it can tip over and flash without
        # touching the root's position and heading.
        self.body = self.root.attachNewNode("body")

        self.hips = self.body.attachNewNode("hips")
        self.hips.setZ(0.95)
        box_node("pelvis", -0.17, 0.17, -0.1, 0.1, -0.12, 0.05, LEATHER_DARK).reparentTo(self.hips)
        # Tabard flaps front and back.
        box_node("tabard_f", -0.13, 0.13, -0.125, -0.1, -0.42, 0.05, CLOTH).reparentTo(self.hips)
        box_node("tabard_b", -0.13, 0.13, 0.1, 0.125, -0.38, 0.05, CLOTH).reparentTo(self.hips)

        self.legs = []
        for side in (-1, 1):
            leg = self.hips.attachNewNode("leg")
            leg.setPos(0.09 * side, 0, -0.08)
            b = MeshBuilder("leg")
            add_box(b, -0.07, 0.07, -0.075, 0.075, -0.45, 0.0, LEATHER)
            add_box(b, -0.065, 0.065, -0.085, 0.07, -0.74, -0.43, STEEL)
            add_box(b, -0.07, 0.07, -0.1, 0.075, -0.5, -0.4, STEEL_DARK)  # knee cop
            add_box(b, -0.075, 0.075, -0.15, 0.08, -0.87, -0.72, LEATHER_DARK)
            b.node().reparentTo(leg)
            self.legs.append(leg)

        self.torso = self.hips.attachNewNode("torso")
        self.torso.setZ(0.05)
        b = MeshBuilder("chest")
        add_box(b, -0.2, 0.2, -0.12, 0.12, 0.0, 0.5, CHAIN)
        add_box(b, -0.18, 0.18, -0.145, -0.1, 0.14, 0.48, STEEL)       # breastplate
        add_box(b, -0.21, 0.21, -0.135, 0.135, 0.0, 0.07, LEATHER)     # belt
        add_box(b, -0.035, 0.035, -0.15, -0.13, 0.005, 0.065, GOLD)    # buckle
        add_box(b, -0.05, 0.05, -0.05, 0.05, 0.5, 0.57, SKIN)          # neck
        add_box(b, -0.27, 0.27, 0.08, 0.17, 0.42, 0.51, CLOAK)         # cloak mantle
        for side in (-1, 1):
            add_box(b, 0.17 * side - 0.035, 0.17 * side + 0.035, -0.16, -0.12, 0.44, 0.5, GOLD)  # clasp
            x = 0.25 * side
            add_box(b, x - 0.1, x + 0.1, -0.13, 0.13, 0.4, 0.52, STEEL)  # pauldron
        b.node().reparentTo(self.torso)

        self.cloak = Cloak(self.torso)

        self.head = self.torso.attachNewNode("head")
        self.head.setZ(0.56)
        b = MeshBuilder("head")
        add_box(b, -0.1, 0.1, -0.1, 0.1, 0.0, 0.22, SKIN)
        add_box(b, -0.115, 0.115, -0.115, 0.115, 0.13, 0.27, STEEL)      # helmet
        add_box(b, -0.115, 0.115, 0.0, 0.115, 0.02, 0.14, STEEL)         # helmet back/cheeks
        add_box(b, -0.115, -0.08, -0.115, 0.0, 0.03, 0.14, STEEL)
        add_box(b, 0.08, 0.115, -0.115, 0.0, 0.03, 0.14, STEEL)
        add_box(b, -0.015, 0.015, -0.135, -0.11, 0.05, 0.17, STEEL_DARK)  # nose guard
        add_box(b, -0.03, 0.03, -0.06, 0.13, 0.27, 0.33, CLOAK)          # crest
        for side in (-1, 1):
            add_box(b, 0.045 * side - 0.02, 0.045 * side + 0.02, -0.105, -0.095, 0.1, 0.125,
                    (0.1, 0.08, 0.06, 1))                                 # eyes
        add_box(b, -0.08, 0.08, -0.11, -0.09, 0.0, 0.06, (0.40, 0.24, 0.12, 1))  # beard
        b.node().reparentTo(self.head)

        # Left arm hangs relaxed.
        self.arm_l = self.torso.attachNewNode("arm_l")
        self.arm_l.setPos(-0.28, 0, 0.44)
        b = MeshBuilder("arm_l")
        add_box(b, -0.06, 0.06, -0.06, 0.06, -0.3, 0.0, CHAIN)
        add_box(b, -0.065, 0.065, -0.065, 0.065, -0.56, -0.29, LEATHER)
        add_box(b, -0.05, 0.05, -0.055, 0.055, -0.66, -0.56, SKIN)
        b.node().reparentTo(self.arm_l)

        # Right arm: forearm bent forward, sword angled down in front.
        self.arm_r = self.torso.attachNewNode("arm_r")
        self.arm_r.setPos(0.28, 0, 0.44)
        box_node("upper_r", -0.06, 0.06, -0.06, 0.06, -0.3, 0.0, CHAIN).reparentTo(self.arm_r)
        self.elbow_r = self.arm_r.attachNewNode("elbow_r")
        self.elbow_r.setZ(-0.29)
        b = MeshBuilder("forearm_r")
        add_box(b, -0.065, 0.065, -0.065, 0.065, -0.27, 0.0, LEATHER)
        add_box(b, -0.055, 0.055, -0.06, 0.06, -0.37, -0.27, SKIN)
        b.node().reparentTo(self.elbow_r)
        self.elbow_r.setP(ELBOW_R_PITCH)

        self.sword = self.elbow_r.attachNewNode("sword")
        self.sword.setZ(-0.32)
        b = MeshBuilder("sword")
        add_box(b, -0.02, 0.02, -0.02, 0.02, -0.08, 0.09, LEATHER_DARK)  # grip
        add_box(b, -0.035, 0.035, -0.035, 0.035, -0.13, -0.08, GOLD)     # pommel
        add_box(b, -0.13, 0.13, -0.025, 0.025, 0.09, 0.125, GOLD)        # crossguard
        add_box(b, -0.032, 0.032, -0.009, 0.009, 0.125, 0.9, BLADE)      # blade
        add_box(b, -0.006, 0.006, -0.011, 0.011, 0.14, 0.85, STEEL_DARK)  # fuller
        for sy in (-1, 1):                                                # tip
            b.tri((-0.032, 0.009 * sy, 0.9), (0.032, 0.009 * sy, 0.9), (0, 0, 1.0),
                  BLADE, (0, sy, 0))
        for sx in (-1, 1):
            b.quad((0.032 * sx, -0.009, 0.9), (0.032 * sx, 0.009, 0.9), (0, 0.0, 1.0),
                   (0, 0.0, 1.0), BLADE, (sx, 0, 0.3))
        b.node().reparentTo(self.sword)

    def animate(self, t, dt, speed, attack=None, dead=0.0):
        """Blend idle and run. speed is ground speed in units/s.

        attack: swing progress 0..1, or None when not swinging.
        dead: 0..1 progress of falling over backward.
        """
        target = min(1.0, speed / RUN_SPEED)
        self.run += (target - self.run) * min(1.0, dt * 10)
        run, idle = self.run, 1.0 - self.run
        # One full cycle (two steps) covers about 1.7 units of ground.
        self.phase = (self.phase + dt * speed * 2 * math.pi / 1.7) % (2 * math.pi)
        stride = math.sin(self.phase)
        bob = abs(math.cos(self.phase))

        breath = math.sin(1.7 * t)
        sway = math.sin(0.6 * t) * idle

        # Hips: idle weight shift, plus bounce and twist while running.
        self.hips.setPos(0.012 * sway, 0, 0.95 - 0.004 * abs(sway) + run * (0.05 * bob - 0.04))
        self.hips.setR(1.2 * sway)
        self.hips.setH(run * 8 * stride)
        for leg, side in zip(self.legs, (-1, 1)):
            # Negative pitch swings a leg forward.
            leg.setP(run * 38 * stride * side)
            leg.setR(-0.4 * sway)

        # Breathing, and a forward lean while running.
        self.torso.setZ(0.05 + 0.006 * breath * idle)
        self.torso.setR(-1.0 * sway)
        self.torso.setP(-1.5 - 0.8 * breath * idle + run * 12)
        self.torso.setH(-run * 14 * stride)

        # Slow look-around when idle, eyes on the road when running.
        self.head.setH(idle * (8 * math.sin(0.37 * t) + 3 * math.sin(0.91 * t + 1.3))
                       + run * 6 * stride)
        self.head.setP(idle * (2.5 * math.sin(0.5 * t) + 1.0 * breath) - run * 8)

        # Arms rise with each breath; when running they swing opposite to the legs.
        self.arm_l.setR(7 + 1.5 * breath * idle + 3 * run)
        self.arm_l.setP(idle * 2 * math.sin(0.8 * t + 1.0) - run * 45 * stride)
        self.arm_r.setR(-8 - 1.5 * breath * idle - 4 * run)
        self.arm_r.setP(ARM_R_PITCH + idle * 1.5 * math.sin(0.9 * t) + run * (25 * stride - 10))
        self.elbow_r.setP(ELBOW_R_PITCH)
        # Sword: blade points forward-down; total chain pitch + sword pitch ~= 153 deg.
        # While running it trails back a little.
        self.sword.setP(153 - ARM_R_PITCH - ELBOW_R_PITCH + 2 * math.sin(1.1 * t + 0.5) * idle
                        + run * 35)
        self.sword.setR(4 * math.sin(0.7 * t) * idle)

        if attack is not None:
            self.apply_swing(attack)

        # Death: topple backward around the feet and settle on the ground.
        f = smoothstep(dead)
        self.body.setP(-84 * f)
        self.body.setZ(0.12 * f)

        self.cloak.update(t, run * (1 - f))

    def apply_swing(self, p):
        arm_p, arm_r, elbow_p, sword_p, torso_h, torso_p = sample_keys(SWING_KEYS, p)
        # Ease in quickly and out slowly so the swing blends with idle/run.
        w = smoothstep(p / 0.12) if p < 0.12 else smoothstep((1 - p) / 0.3) if p > 0.7 else 1.0
        self.arm_r.setP(lerp(self.arm_r.getP(), arm_p, w))
        self.arm_r.setR(lerp(self.arm_r.getR(), arm_r, w))
        self.elbow_r.setP(lerp(ELBOW_R_PITCH, elbow_p, w))
        self.sword.setP(lerp(self.sword.getP(), sword_p, w))
        self.sword.setR(lerp(self.sword.getR(), 0, w))
        self.torso.setH(lerp(self.torso.getH(), torso_h, w))
        self.torso.setP(lerp(self.torso.getP(), torso_p, w))
