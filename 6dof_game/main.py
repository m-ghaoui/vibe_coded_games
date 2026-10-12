"""A tiny 6DoF Quake/Descent-style shooter built with Panda3D.

Everything (level, textures, models) is generated procedurally at startup.
Run with:  .venv/bin/python main.py
"""

import math
import random
import sys

from panda3d.core import (
    loadPrcFileData,
    AmbientLight,
    BitMask32,
    CardMaker,
    CollisionHandlerPusher,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionPolygon,
    CollisionSegment,
    CollisionSphere,
    CollisionTraverser,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    KeyboardButton,
    MouseButton,
    PerspectiveLens,
    PNMImage,
    Point3,
    PointLight,
    Quat,
    SamplerState,
    Spotlight,
    Texture,
    Vec3,
    Vec4,
    WindowProperties,
)

SCREENSHOT_MODE = "--screenshot" in sys.argv

loadPrcFileData("", """
window-title Quake6 - 6DoF Panda3D shooter
win-size 1280 720
sync-video true
show-frame-rate-meter true
framebuffer-multisample 1
multisamples 4
""")
if SCREENSHOT_MODE:
    loadPrcFileData("", "window-type offscreen\nsync-video false")

from direct.showbase.ShowBase import ShowBase  # noqa: E402  (after prc config)
from direct.gui.OnscreenText import OnscreenText  # noqa: E402
from direct.filter.CommonFilters import CommonFilters  # noqa: E402

# --- Tunables ---------------------------------------------------------------
CELL = 4.0                    # world units per level grid cell
NX, NY, NZ = 24, 24, 8        # level grid dimensions
MOUSE_SENS = 0.12             # degrees per pixel
ROLL_SPEED = 120.0            # degrees per second
ACCEL = 70.0
DAMPING = 4.0                 # higher = less floaty
BOOST = 2.0
PLAYER_RADIUS = 1.2
BOLT_SPEED = 70.0
BOLT_LIFE = 2.0
FIRE_COOLDOWN = 0.14
LIGHT_POOL_SIZE = 10
SHADOW_SIZE = 1024             # per cube-map face for each lamp
PLAYER_HEALTH = 100
REGEN_DELAY = 5.0              # seconds without damage before health regenerates
REGEN_RATE = 4.0               # health per second

# Drone AI
DRONE_HEALTH = 3
DRONE_ACCEL = 30.0
DRONE_TURN_RATE = 5.0
SIGHT_RANGE = 70.0
SIGHT_INTERVAL = 0.2           # seconds between line-of-sight checks
REACTION_TIME = 0.7            # delay between spotting the player and first shot
PREFERRED_DIST = 16.0
SEARCH_TIME = 6.0
ALERT_RADIUS = 30.0            # drones this close to a hit drone come looking
ENEMY_FIRE_INTERVAL = (1.3, 2.4)
ENEMY_TELEGRAPH = 0.4          # eye glows this long before a shot
ENEMY_BOLT_SPEED = 38.0
ENEMY_SPREAD = 0.04            # radians-ish of aim error
ENEMY_DAMAGE = 8

MASK_WALL = BitMask32.bit(1)
MASK_DRONE = BitMask32.bit(2)
MASK_PLAYER = BitMask32.bit(4)

CAM_MAIN = BitMask32.bit(0)     # camera masks: what the player sees...
CAM_SHADOW = BitMask32.bit(3)   # ...vs what shadow-map cameras render

DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def make_unlit(np):
    """Self-illuminated object: ignores lighting and casts no shadows."""
    np.setLightOff()
    np.hide(CAM_SHADOW)


def wp(x, y, z):
    """Grid coordinates (in cells, may be fractional) -> world position."""
    return Point3(x * CELL, y * CELL, z * CELL)


# --- Geometry helpers -------------------------------------------------------
class MeshBuilder:
    def __init__(self, name):
        self.name = name
        self.vdata = GeomVertexData(name, GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
        self.vw = GeomVertexWriter(self.vdata, "vertex")
        self.nw = GeomVertexWriter(self.vdata, "normal")
        self.tw = GeomVertexWriter(self.vdata, "texcoord")
        self.tris = GeomTriangles(Geom.UHStatic)
        self.count = 0

    def quad(self, corners, normal, uvs):
        """Add a quad; fixes winding so it faces `normal`. Returns corners in CCW order."""
        a, b, c = corners[0], corners[1], corners[2]
        if (b - a).cross(c - a).dot(normal) < 0:
            corners = corners[::-1]
            uvs = uvs[::-1]
        for p, uv in zip(corners, uvs):
            self.vw.addData3(p)
            self.nw.addData3(normal)
            self.tw.addData2(uv)
        i = self.count
        self.tris.addVertices(i, i + 1, i + 2)
        self.tris.addVertices(i, i + 2, i + 3)
        self.count += 4
        return corners

    def node(self):
        geom = Geom(self.vdata)
        geom.addPrimitive(self.tris)
        gn = GeomNode(self.name)
        gn.addGeom(geom)
        return gn


def face_axes(n):
    """Two tangent axes for an axis-aligned face normal."""
    if n.x != 0:
        return Vec3(0, 1, 0), Vec3(0, 0, 1)
    if n.y != 0:
        return Vec3(1, 0, 0), Vec3(0, 0, 1)
    return Vec3(1, 0, 0), Vec3(0, 1, 0)


def face_corners(center, u, v, hu, hv):
    return [center - u * hu - v * hv, center + u * hu - v * hv,
            center + u * hu + v * hv, center - u * hu + v * hv]


def make_box(name, hx, hy, hz):
    """Outward-facing box centered on the origin."""
    mb = MeshBuilder(name)
    half = Vec3(hx, hy, hz)
    for d in DIRS:
        n = Vec3(*d)
        u, v = face_axes(n)
        center = Point3(n.x * hx, n.y * hy, n.z * hz)
        mb.quad(face_corners(center, u, v, abs(u.dot(half)), abs(v.dot(half))),
                n, [(0, 0), (1, 0), (1, 1), (0, 1)])
    return mb.node()


# --- Procedural textures ----------------------------------------------------
def make_texture(kind, seed, size=128):
    rng = random.Random(seed)
    img = PNMImage(size, size)
    for y in range(size):
        for x in range(size):
            noise = 0.8 + 0.3 * rng.random()
            if kind == "brick":
                row = y // 16
                bx = (x + (16 if row % 2 else 0)) % 32
                mortar = (y % 16) < 2 or bx < 2
                base = (0.18, 0.15, 0.12) if mortar else (0.48, 0.34, 0.22)
                shade = 1.0 if mortar else 0.85 + 0.15 * math.sin(row * 1.7 + (x + 16 * row) // 32)
            elif kind == "floor":
                edge = (x % 32) < 2 or (y % 32) < 2
                base = (0.16, 0.16, 0.14) if edge else (0.36, 0.36, 0.31)
                shade = 1.0 - 0.15 * (((x // 32) + (y // 32)) % 2)
            else:  # ceiling: dark riveted metal
                rivet = (x % 32 - 16) ** 2 + (y % 32 - 16) ** 2 < 5
                seam = (x % 64) < 2 or (y % 64) < 2
                base = (0.12, 0.1, 0.09) if seam else (0.3, 0.27, 0.24)
                shade = 1.4 if rivet else 1.0
            img.setXel(x, y, *(min(1.0, c * noise * shade) for c in base))
    tex = Texture(kind)
    tex.load(img)
    tex.setMagfilter(SamplerState.FT_nearest)  # crunchy retro look
    tex.setMinfilter(SamplerState.FT_linear_mipmap_linear)
    tex.setAnisotropicDegree(4)
    return tex


# --- Level ------------------------------------------------------------------
class Level:
    """A voxel-ish grid: every cell is solid or empty. Faces between empty and
    solid cells become inward-facing, textured, collidable polygons."""

    def __init__(self):
        self.solid = [[[True] * NZ for _ in range(NY)] for _ in range(NX)]

    def carve(self, x0, y0, z0, x1, y1, z1, value=False):
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    self.solid[x][y][z] = value

    def fill(self, *box):
        self.carve(*box, value=True)

    def is_solid(self, x, y, z):
        if not (0 <= x < NX and 0 <= y < NY and 0 <= z < NZ):
            return True
        return self.solid[x][y][z]

    def build(self, parent, textures):
        builders = {k: MeshBuilder("level-" + k) for k in textures}
        root = parent.attachNewNode("level")
        for x in range(NX):
            # one collision node per slice so traversal can cull by bounds
            cnode = CollisionNode(f"level-col-{x}")
            cnode.setIntoCollideMask(MASK_WALL)
            cnode.setFromCollideMask(BitMask32.allOff())
            for y in range(NY):
                for z in range(NZ):
                    if self.solid[x][y][z]:
                        continue
                    center = Point3((x + 0.5) * CELL, (y + 0.5) * CELL, (z + 0.5) * CELL)
                    for d in DIRS:
                        if not self.is_solid(x + d[0], y + d[1], z + d[2]):
                            continue
                        n = Vec3(*d)
                        kind = "floor" if d[2] < 0 else "ceil" if d[2] > 0 else "brick"
                        u, v = face_axes(n)
                        corners = face_corners(center + n * (CELL / 2), u, v, CELL / 2, CELL / 2)
                        uvs = [(p.dot(u) / CELL, p.dot(v) / CELL) for p in corners]
                        corners = builders[kind].quad(corners, -n, uvs)
                        cnode.addSolid(CollisionPolygon(*corners))
            if cnode.getNumSolids():
                root.attachNewNode(cnode)
        for kind, mb in builders.items():
            np = root.attachNewNode(mb.node())
            np.setTexture(textures[kind])
        return root


def build_layout():
    lv = Level()
    # Central hall with four pillars and a floating bridge
    lv.carve(8, 8, 0, 16, 16, 4)
    for px, py in ((10, 10), (13, 10), (10, 13), (13, 13)):
        lv.fill(px, py, 0, px + 1, py + 1, 4)
    lv.fill(8, 11, 2, 16, 13, 3)
    # North room
    lv.carve(11, 16, 0, 13, 19, 2)
    lv.carve(8, 19, 0, 16, 23, 3)
    lv.fill(9, 21, 0, 10, 22, 1)
    lv.fill(14, 20, 0, 15, 21, 1)
    # South room (low and wide)
    lv.carve(11, 5, 0, 13, 8, 1)
    lv.carve(9, 1, 0, 15, 5, 2)
    # East room, reached by a raised corridor
    lv.carve(16, 11, 1, 19, 13, 2)
    lv.carve(19, 8, 0, 23, 16, 4)
    lv.fill(19, 9, 0, 20, 10, 1)
    lv.fill(21, 14, 0, 22, 15, 2)
    # West: corridor into a tall vertical shaft leading to an upper room
    lv.carve(5, 11, 0, 8, 13, 1)
    lv.carve(3, 10, 0, 6, 14, 7)
    lv.carve(4, 8, 5, 6, 10, 6)
    lv.carve(1, 1, 5, 9, 9, 7)
    lv.fill(2, 2, 5, 3, 3, 6)
    lv.fill(6, 6, 5, 7, 7, 7)

    lights = [  # (pos, color, flicker)
        (wp(9, 9, 3.6), (1.5, 0.9, 0.45), True),
        (wp(15, 15, 3.6), (1.5, 0.9, 0.45), True),
        (wp(12, 9, 0.6), (0.3, 0.5, 1.4), False),
        (wp(12, 21.5, 2.6), (1.6, 0.25, 0.2), True),
        (wp(12, 3, 1.6), (0.3, 1.3, 0.4), False),
        (wp(21, 12, 3.6), (0.9, 0.4, 1.5), False),
        (wp(4.5, 12, 6.6), (0.6, 1.2, 1.5), False),
        (wp(5, 5, 6.6), (1.5, 1.3, 0.5), True),
    ]
    drones = [wp(12, 20.5, 1), wp(20, 10, 2), wp(21.5, 14, 2.8), wp(12, 3, 0.8),
              wp(4.5, 12, 4), wp(5, 4, 5.6), wp(9.5, 15, 3.4)]
    spawn = wp(12, 9, 1)
    return lv, lights, drones, spawn


# --- Runtime objects --------------------------------------------------------
class LightPool:
    """Fixed set of always-enabled point lights for transient effects.
    Toggling lights on/off would force the auto-shader to recompile, so we
    keep them enabled and just set the color to black when unused."""

    def __init__(self, render, n):
        self.free = []
        for i in range(n):
            pl = PointLight(f"pool-{i}")
            pl.setAttenuation((1, 0, 0.02))
            np = render.attachNewNode(pl)
            render.setLight(np)
            self.release(np)

    def acquire(self, color):
        if not self.free:
            return None
        np = self.free.pop()
        np.node().setColor(color)
        return np

    def release(self, np):
        np.node().setColor((0, 0, 0, 1))
        self.free.append(np)


def nlerp(q0, q1, a):
    """Normalized blend between two rotations (shortest path)."""
    if q0.dot(q1) < 0:
        q1 = Quat(-q1.getR(), -q1.getI(), -q1.getJ(), -q1.getK())
    q = Quat(*(c0 + (c1 - c0) * a for c0, c1 in zip(q0, q1)))
    q.normalize()
    return q


class Drone:
    """Hovering sentry with a small state machine:
    idle   - drift around its home position
    attack - player visible: keep preferred range, strafe, lead shots
    search - lost sight (or alerted): fly to the last known position, then give up
    """

    EYE_COLOR = Vec4(2.0, 0.2, 0.1, 1)
    CHARGE_COLOR = Vec4(5.0, 1.8, 0.6, 1)

    def __init__(self, game, pos):
        self.game = game
        self.home = Point3(pos)
        self.phase = random.uniform(0, math.tau)
        self.vel = Vec3(0, 0, 0)
        self.health = DRONE_HEALTH
        self.state = "idle"
        self.sees_player = False
        self.last_seen = None
        self.search_timer = 0.0
        self.sight_timer = random.uniform(0, SIGHT_INTERVAL)  # stagger checks across drones
        self.reaction = 0.0
        self.fire_timer = random.uniform(*ENEMY_FIRE_INTERVAL)
        self.wander_target = Point3(pos)
        self.wander_timer = 0.0
        self.strafe_dir = random.choice((-1, 1))
        self.strafe_timer = random.uniform(1, 3)
        self.hit_flash = 0.0

        self.np = game.render.attachNewNode("drone")
        self.np.setPos(pos)
        self.np.setH(random.uniform(0, 360))
        self.body = self.np.attachNewNode(make_box("drone-body", 1.0, 1.0, 1.0))
        self.body.setTexture(game.textures["ceil"])
        self.body.setColor(0.8, 0.8, 0.9, 1)
        self.eye = self.np.attachNewNode(make_box("drone-eye", 0.35, 0.1, 0.35))
        self.eye.setY(1.05)  # +Y is forward, so lookAt() aims the eye
        make_unlit(self.eye)
        self.eye.setColor(self.EYE_COLOR)
        for sx in (-1, 1):
            fin = self.np.attachNewNode(make_box("fin", 0.1, 0.7, 0.6))
            fin.setX(sx * 1.1)
            fin.setTexture(game.textures["brick"])

        cnode = CollisionNode("drone")
        cnode.addSolid(CollisionSphere(0, 0, 0, 1.4))
        cnode.setIntoCollideMask(MASK_DRONE)
        cnode.setFromCollideMask(MASK_WALL | MASK_DRONE)
        cnode.setPythonTag("drone", self)
        self.cnp = self.np.attachNewNode(cnode)
        game.drone_pusher.addCollider(self.cnp, self.np)
        game.cTrav.addCollider(self.cnp, game.drone_pusher)

    def alert(self, pos):
        """Something happened at `pos` (we got shot, a friend got shot): go look."""
        if self.state != "attack":
            self.state = "search"
            self.last_seen = Point3(pos)
            self.search_timer = SEARCH_TIME

    def take_hit(self, bolt_vel):
        self.health -= 1
        self.hit_flash = 0.12
        self.vel += bolt_vel.normalized() * 8
        self.alert(self.game.player.getPos())
        return self.health <= 0

    def update(self, dt, t):
        game = self.game
        pos = self.np.getPos()
        player_pos = game.player.getPos()
        to_player = player_pos - pos
        dist = max(to_player.length(), 0.001)

        self.sight_timer -= dt
        if self.sight_timer <= 0:
            self.sight_timer = SIGHT_INTERVAL
            self.sees_player = (not game.player_dead and dist < SIGHT_RANGE
                                and game.line_of_sight(pos, player_pos))
            if self.sees_player and self.state != "attack":
                if self.state == "idle":  # surprised: hesitate before the first shot
                    self.reaction = REACTION_TIME
                self.state = "attack"
            elif not self.sees_player and self.state == "attack":
                self.state = "search"
                self.search_timer = SEARCH_TIME

        desired = Vec3(0, 0, 0)
        look_target = None
        if self.state == "attack":
            self.last_seen = Point3(player_pos)
            direction = to_player / dist
            side = direction.cross(Vec3(0, 0, 1))
            if side.lengthSquared() < 0.01:
                side = Vec3(1, 0, 0)
            side.normalize()
            self.strafe_timer -= dt
            if self.strafe_timer <= 0:
                self.strafe_dir *= -1
                self.strafe_timer = random.uniform(1.0, 3.0)
            approach = max(-1.0, min(1.0, (dist - PREFERRED_DIST) / 4))
            desired = (direction * approach + side * (self.strafe_dir * 0.7)
                       + Vec3(0, 0, 0.5 * math.sin(t * 1.3 + self.phase)))
            # Lead the target: aim where the player will be when the bolt arrives
            look_target = player_pos + game.velocity * (dist / ENEMY_BOLT_SPEED) * 0.7
        elif self.state == "search":
            self.search_timer -= dt
            off = self.last_seen - pos
            if off.length() < 3 or self.search_timer <= 0:
                self.state = "idle"
                self.wander_timer = 0
            else:
                desired = off.normalized()
                look_target = self.last_seen
        else:
            self.wander_timer -= dt
            if self.wander_timer <= 0 or (self.wander_target - pos).length() < 1:
                self.wander_target = self.home + Vec3(random.uniform(-3, 3), random.uniform(-3, 3),
                                                      random.uniform(-1.5, 1.5))
                self.wander_timer = random.uniform(2, 4)
            off = self.wander_target - pos
            desired = off * 0.15
            if off.length() > 1:
                look_target = self.wander_target

        if desired.lengthSquared() > 1:
            desired.normalize()
        self.vel += desired * DRONE_ACCEL * dt
        self.vel *= math.exp(-2.5 * dt)
        self.np.setFluidPos(pos + self.vel * dt)
        if look_target is not None:
            game.aim_helper.setPos(pos)
            game.aim_helper.lookAt(look_target)
            a = 1 - math.exp(-DRONE_TURN_RATE * dt)
            self.np.setQuat(nlerp(self.np.getQuat(), game.aim_helper.getQuat(), a))

        self.update_weapon(dt, look_target)
        self.hit_flash = max(0.0, self.hit_flash - dt)
        self.body.setColor((3, 3, 3, 1) if self.hit_flash > 0 else (0.8, 0.8, 0.9, 1))

    def update_weapon(self, dt, aim_point):
        charge = 0.0
        if self.state == "attack" and self.sees_player:
            self.reaction -= dt
            self.fire_timer -= dt
            if self.reaction <= 0 and self.fire_timer < ENEMY_TELEGRAPH:
                charge = 1 - max(0.0, self.fire_timer) / ENEMY_TELEGRAPH
            if self.reaction <= 0 and self.fire_timer <= 0:
                muzzle = self.np.getPos() + self.np.getQuat().getForward() * 1.6
                aim = aim_point - muzzle
                aim.normalize()
                # Only shoot when roughly facing the target
                if aim.dot(self.np.getQuat().getForward()) > 0.95:
                    spread = Vec3(*(random.gauss(0, ENEMY_SPREAD) for _ in range(3)))
                    self.game.spawn_bolt(muzzle, (aim + spread).normalized(), enemy=True)
                    self.fire_timer = random.uniform(*ENEMY_FIRE_INTERVAL)
        else:
            # keep some cooldown so a drone doesn't fire instantly on re-acquire
            self.fire_timer = max(self.fire_timer, ENEMY_TELEGRAPH + 0.2)
        self.eye.setColor(self.EYE_COLOR + (self.CHARGE_COLOR - self.EYE_COLOR) * charge)
        self.eye.setScale(1 + 0.6 * charge)

    def destroy(self):
        self.game.cTrav.removeCollider(self.cnp)
        self.game.drone_pusher.removeCollider(self.cnp)
        self.np.removeNode()


class Bolt:
    def __init__(self, game, pos, direction, enemy=False):
        self.game = game
        self.enemy = enemy
        self.vel = direction * (ENEMY_BOLT_SPEED if enemy else BOLT_SPEED)
        self.age = 0.0
        self.prev = Point3(pos)
        color = (2.2, 0.35, 0.15) if enemy else (1.6, 1.2, 0.4)
        self.np = game.render.attachNewNode("bolt")
        self.np.setPos(pos)
        self.np.lookAt(pos + direction)
        size = 0.14 if enemy else 0.1
        model = self.np.attachNewNode(make_box("bolt", size, 0.45, size))
        make_unlit(model)
        model.setColor(*color, 1)
        self.light = game.light_pool.acquire((*(c * 0.75 for c in color), 1))
        self.cnode = CollisionNode("bolt-seg")
        self.cnode.setFromCollideMask(MASK_WALL | (MASK_PLAYER if enemy else MASK_DRONE))
        self.cnode.setIntoCollideMask(BitMask32.allOff())
        self.cnode.setPythonTag("bolt", self)
        self.cnp = game.render.attachNewNode(self.cnode)
        game.bolt_trav.addCollider(self.cnp, game.bolt_queue)

    def advance(self, dt):
        """Move and set the swept segment for this frame's hit test."""
        self.age += dt
        prev = self.np.getPos()
        new = prev + self.vel * dt
        self.cnode.clearSolids()
        self.cnode.addSolid(CollisionSegment(prev, new))
        self.prev = prev
        self.np.setPos(new)
        if self.light:
            self.light.setPos(new)

    def destroy(self):
        if self.light:
            self.game.light_pool.release(self.light)
        self.game.bolt_trav.removeCollider(self.cnp)
        self.cnp.removeNode()
        self.np.removeNode()


class Flash:
    """Short-lived explosion: a pooled light plus a shrinking glowing cube."""

    def __init__(self, game, pos, color, size, duration):
        self.game, self.color, self.duration, self.age = game, Vec4(*color, 1), duration, 0.0
        self.light = game.light_pool.acquire(self.color)
        if self.light:
            self.light.setPos(pos)
        self.model = game.render.attachNewNode(make_box("flash", size, size, size))
        self.model.setPos(pos)
        make_unlit(self.model)
        self.model.setColor(self.color * 1.5)
        self.model.setTransparency(True)

    def update(self, dt):
        self.age += dt
        k = max(0.0, 1.0 - self.age / self.duration)
        if self.light:
            self.light.node().setColor(self.color * k * 2)
        self.model.setScale(0.3 + k)
        self.model.setAlphaScale(k)
        self.model.setHpr(self.age * 400, self.age * 300, 0)
        return k > 0

    def destroy(self):
        if self.light:
            self.game.light_pool.release(self.light)
        self.model.removeNode()


# --- Game -------------------------------------------------------------------
class Game(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.02, 0.015, 0.01, 1)
        self.camLens.setFov(95)
        self.camLens.setNear(0.1)
        self.cam.node().setCameraMask(CAM_MAIN)

        self.textures = {k: make_texture(k, i) for i, k in enumerate(("brick", "floor", "ceil"))}
        level, light_defs, drone_positions, spawn = build_layout()
        level.build(self.render, self.textures)
        self.drone_positions = drone_positions
        self.spawn_point = spawn

        self.setup_lights(light_defs)
        self.setup_player(spawn)
        self.light_pool = LightPool(self.render, LIGHT_POOL_SIZE)

        fog = Fog("fog")
        fog.setColor(0.02, 0.015, 0.01)
        fog.setExpDensity(0.012)
        self.render.setFog(fog)
        self.render.setShaderAuto()

        self.filters = CommonFilters(self.win, self.cam)
        self.filters.setBloom(blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.7, maxtrigger=1.0,
                              desat=0.3, intensity=0.8, size="medium")

        self.bolt_trav = CollisionTraverser("bolts")
        self.bolt_queue = CollisionHandlerQueue()
        self.los_trav = CollisionTraverser("line-of-sight")
        self.los_queue = CollisionHandlerQueue()
        self.los_node = CollisionNode("los")
        self.los_node.setFromCollideMask(MASK_WALL)
        self.los_node.setIntoCollideMask(BitMask32.allOff())
        self.los_trav.addCollider(self.render.attachNewNode(self.los_node), self.los_queue)
        self.aim_helper = self.render.attachNewNode("aim-helper")
        self.drone_pusher = CollisionHandlerPusher()

        self.bolts, self.flashes, self.drones = [], [], []
        self.setup_hud()
        self.restart_game()

        self.capture_mouse()
        self.accept("escape", sys.exit)
        self.accept("f", self.toggle_headlight)
        self.accept("r", self.restart_game)
        self.accept("g", self.toggle_shadows)
        self.taskMgr.add(self.update, "update")

    # -- setup --
    def setup_lights(self, light_defs):
        amb = AmbientLight("ambient")
        amb.setColor((0.04, 0.035, 0.045, 1))
        self.render.setLight(self.render.attachNewNode(amb))
        self.flicker_lights = []
        self.shadow_lights = []
        for i, (pos, color, flicker) in enumerate(light_defs):
            pl = PointLight(f"lamp-{i}")
            color = tuple(c * 1.8 for c in color)
            pl.setColor((*color, 1))
            pl.setAttenuation((1, 0, 0.008))
            pl.setCameraMask(CAM_SHADOW)
            pl.setMaxDistance(60)
            pl.setShadowCaster(True, SHADOW_SIZE, SHADOW_SIZE)
            self.shadow_lights.append(pl)
            np = self.render.attachNewNode(pl)
            np.setPos(pos)
            self.render.setLight(np)
            lamp = np.attachNewNode(make_box("lamp", 0.35, 0.35, 0.35))
            make_unlit(lamp)
            lamp.setColor(*color, 1)
            if flicker:
                self.flicker_lights.append((pl, Vec4(*color, 1), random.uniform(0, 100)))

    def setup_player(self, spawn):
        self.player = self.render.attachNewNode("player")
        self.player.setPos(spawn)
        self.camera.reparentTo(self.player)
        self.camera.setPos(0, 0, 0)
        self.velocity = Vec3(0, 0, 0)

        cnode = CollisionNode("player")
        cnode.addSolid(CollisionSphere(0, 0, 0, PLAYER_RADIUS))
        cnode.setFromCollideMask(MASK_WALL | MASK_DRONE)
        cnode.setIntoCollideMask(MASK_PLAYER)
        cnode.setPythonTag("player", True)
        cnp = self.player.attachNewNode(cnode)
        self.cTrav = CollisionTraverser("world")
        self.cTrav.setRespectPrevTransform(True)
        pusher = CollisionHandlerPusher()
        pusher.addCollider(cnp, self.player)
        self.cTrav.addCollider(cnp, pusher)

        # Headlight (spotlight) and muzzle flash ride along with the camera
        spot = Spotlight("headlight")
        lens = PerspectiveLens()
        lens.setFov(55)
        spot.setLens(lens)
        spot.setExponent(8)
        spot.setAttenuation((1, 0, 0.003))
        self.headlight_color = Vec4(0.9, 0.9, 0.8, 1)
        spot.setColor(self.headlight_color)
        self.headlight = self.camera.attachNewNode(spot)
        self.headlight.setPos(0.3, 0, -0.3)
        self.render.setLight(self.headlight)
        spot.setCameraMask(CAM_SHADOW)
        spot.setShadowCaster(True, 2048, 2048)
        lens.setNearFar(0.5, 80)
        self.shadow_lights.append(spot)
        self.shadows_on = True
        self.headlight_on = True

        muzzle = PointLight("muzzle")
        muzzle.setAttenuation((1, 0, 0.02))
        muzzle.setColor((0, 0, 0, 1))
        self.muzzle = self.camera.attachNewNode(muzzle)
        self.muzzle.setPos(0, 1.5, -0.3)
        self.render.setLight(self.muzzle)
        self.muzzle_level = 0.0

    def setup_hud(self):
        OnscreenText(text="+", pos=(0, -0.02), scale=0.07, fg=(1, 0.9, 0.6, 0.8), mayChange=False)
        self.hud = OnscreenText(text="", pos=(-1.3, 0.9), scale=0.055, align=0,
                                fg=(1, 0.8, 0.4, 1), shadow=(0, 0, 0, 1), mayChange=True)
        self.health_text = OnscreenText(text="", pos=(-1.3, -0.85), scale=0.09, align=0,
                                        fg=(1, 0.8, 0.4, 1), shadow=(0, 0, 0, 1), mayChange=True)
        self.banner = OnscreenText(text="", pos=(0, 0.15), scale=0.1, fg=(1, 0.25, 0.15, 1),
                                   shadow=(0, 0, 0, 1), mayChange=True)
        if not SCREENSHOT_MODE:
            OnscreenText(text="Mouse: look  WASD: move  Space/Ctrl: up/down  Q/E: roll  "
                              "Shift: boost  LMB: fire  F: headlight  G: shadows  R: restart  Esc: quit",
                         pos=(0, -0.95), scale=0.04, fg=(0.8, 0.8, 0.8, 0.8), shadow=(0, 0, 0, 1))
        cm = CardMaker("damage-overlay")
        cm.setFrameFullscreenQuad()
        self.damage_overlay = self.render2d.attachNewNode(cm.generate())
        self.damage_overlay.setTransparency(True)
        self.damage_overlay.setColor(1, 0, 0, 0)
        self.damage_level = 0.0
        self.shake = 0.0

    def capture_mouse(self):
        if SCREENSHOT_MODE:
            self.mouse_ready = False
            return
        props = WindowProperties()
        props.setCursorHidden(True)
        props.setMouseMode(WindowProperties.M_confined)
        self.win.requestProperties(props)
        self.mouse_ready = False

    def restart_game(self):
        for obj in self.drones + self.bolts + self.flashes:
            obj.destroy()
        self.bolts, self.flashes = [], []
        self.drones = [Drone(self, p) for p in self.drone_positions]
        self.player.setPos(self.spawn_point)
        self.player.setHpr(0, 0, 0)
        self.camera.setHpr(0, 0, 0)
        self.velocity = Vec3(0, 0, 0)
        self.health = PLAYER_HEALTH
        self.player_dead = False
        self.since_damage = REGEN_DELAY
        self.score = 0
        self.fire_timer = 0.0
        self.banner.setText("")
        self.update_hud()

    def update_hud(self):
        msg = f"SCORE {self.score}    DRONES {len(self.drones)}"
        self.hud.setText(msg)
        hp = max(0, math.ceil(self.health))
        self.health_text.setText(f"+{hp}")
        self.health_text.setFg((1, 0.25, 0.15, 1) if hp <= 30 else (1, 0.8, 0.4, 1))
        if not self.drones and not self.player_dead:
            self.banner.setFg((1, 0.85, 0.3, 1))
            self.banner.setText("ALL CLEAR!\npress R to play again")

    def line_of_sight(self, a, b):
        self.los_node.clearSolids()
        self.los_node.addSolid(CollisionSegment(a, b))
        self.los_trav.traverse(self.render)
        return self.los_queue.getNumEntries() == 0

    def damage_player(self, amount, push):
        if self.player_dead:
            return
        self.health -= amount
        self.since_damage = 0.0
        self.damage_level = min(0.6, self.damage_level + 0.35)
        self.shake = min(1.0, self.shake + 0.5)
        self.velocity += push.normalized() * 6
        player_pos = self.player.getPos()
        for d in self.drones:  # nearby drones hear the commotion
            if (d.np.getPos() - player_pos).length() < ALERT_RADIUS:
                d.alert(player_pos)
        if self.health <= 0:
            self.player_dead = True
            self.banner.setFg((1, 0.25, 0.15, 1))
            self.banner.setText("YOU DIED\npress R to restart")
        self.update_hud()

    def kill_drone(self, drone, point):
        self.drones.remove(drone)
        drone.destroy()
        self.score += 100
        self.flashes.append(Flash(self, point, (1.5, 0.6, 0.2), 1.4, 0.7))
        for d in self.drones:
            if (d.np.getPos() - point).length() < ALERT_RADIUS:
                d.alert(self.player.getPos())
        self.update_hud()

    def toggle_shadows(self):
        self.shadows_on = not self.shadows_on
        for light in self.shadow_lights:
            size = light.getShadowBufferSize()
            light.setShadowCaster(self.shadows_on, size.x, size.y)

    def toggle_headlight(self):
        self.headlight_on = not self.headlight_on
        self.headlight.node().setColor(self.headlight_color if self.headlight_on else Vec4(0, 0, 0, 1))

    # -- per-frame --
    def update(self, task):
        dt = min(self.clock.getDt(), 0.05)
        t = task.time
        if not self.player_dead:
            self.update_look(dt)
            self.update_movement(dt)
            self.update_weapon(dt)
            self.update_health(dt)
        else:
            # slowly tumble while drifting to a stop
            self.player.setHpr(self.player, 0, 0, 40 * dt)
            self.velocity *= math.exp(-3 * dt)
            self.player.setFluidPos(self.player.getPos() + self.velocity * dt)
        self.update_bolts(dt)
        self.flashes = [f for f in self.flashes if f.update(dt) or f.destroy()]
        for d in self.drones:
            d.update(dt, t)
        self.update_damage_fx(dt)
        for pl, color, seed in self.flicker_lights:
            k = 0.85 + 0.1 * math.sin(t * 13 + seed) + 0.05 * math.sin(t * 31 + seed * 2)
            pl.setColor(color * k)
        return task.cont

    def update_health(self, dt):
        self.since_damage += dt
        if self.since_damage > REGEN_DELAY and self.health < PLAYER_HEALTH:
            before = math.ceil(self.health)
            self.health = min(PLAYER_HEALTH, self.health + REGEN_RATE * dt)
            if math.ceil(self.health) != before:
                self.update_hud()

    def update_damage_fx(self, dt):
        self.damage_level = max(0.0, self.damage_level - dt * 1.5)
        dead_tint = 0.35 if self.player_dead else 0.0
        self.damage_overlay.setColor(1, 0, 0, max(self.damage_level, dead_tint))
        self.shake = max(0.0, self.shake - dt * 3)
        s = self.shake * 2.5
        self.camera.setHpr(random.uniform(-s, s), random.uniform(-s, s), random.uniform(-s, s))

    def is_down(self, key):
        return self.mouseWatcherNode is not None and self.mouseWatcherNode.isButtonDown(key)

    def update_look(self, dt):
        roll = 0.0
        if self.is_down(KeyboardButton.asciiKey("q")):
            roll -= ROLL_SPEED * dt
        if self.is_down(KeyboardButton.asciiKey("e")):
            roll += ROLL_SPEED * dt
        dh = dp = 0.0
        if not SCREENSHOT_MODE:
            md = self.win.getPointer(0)
            cx, cy = self.win.getXSize() // 2, self.win.getYSize() // 2
            if md.getInWindow():
                if self.mouse_ready:
                    dh = -(md.getX() - cx) * MOUSE_SENS
                    dp = -(md.getY() - cy) * MOUSE_SENS
                self.mouse_ready = self.win.movePointer(0, cx, cy)
        # Rotating relative to ourselves composes local rotations: true 6DoF, no gimbal lock
        if dh or dp or roll:
            self.player.setHpr(self.player, dh, dp, roll)

    def update_movement(self, dt):
        move = Vec3(0, 0, 0)
        keys = {"w": (0, 1, 0), "s": (0, -1, 0), "a": (-1, 0, 0), "d": (1, 0, 0)}
        for k, v in keys.items():
            if self.is_down(KeyboardButton.asciiKey(k)):
                move += Vec3(*v)
        if self.is_down(KeyboardButton.space()):
            move.z += 1
        if self.is_down(KeyboardButton.control()):
            move.z -= 1
        if move.lengthSquared() > 0:
            move.normalize()
        accel = ACCEL * (BOOST if self.is_down(KeyboardButton.shift()) else 1.0)
        world_move = self.player.getQuat().xform(move)
        self.velocity += world_move * accel * dt
        self.velocity *= math.exp(-DAMPING * dt)
        self.player.setFluidPos(self.player.getPos() + self.velocity * dt)

    def update_weapon(self, dt):
        self.fire_timer -= dt
        if self.is_down(MouseButton.one()) and self.fire_timer <= 0 and not SCREENSHOT_MODE:
            self.fire()
        self.muzzle_level = max(0.0, self.muzzle_level - dt * 12)
        self.muzzle.node().setColor(Vec4(1.5, 1.0, 0.4, 1) * self.muzzle_level)

    def fire(self, direction=None):
        self.fire_timer = FIRE_COOLDOWN
        q = self.player.getQuat()
        forward = q.getForward() if direction is None else direction
        start = self.player.getPos() + forward * 0.8 + q.getUp() * -0.25
        self.spawn_bolt(start, forward)
        self.muzzle_level = 1.0

    def spawn_bolt(self, pos, direction, enemy=False):
        self.bolts.append(Bolt(self, pos, direction, enemy))

    def update_bolts(self, dt):
        if not self.bolts:
            return
        for b in self.bolts:
            b.advance(dt)
        self.bolt_trav.traverse(self.render)
        nearest = {}
        for entry in self.bolt_queue.getEntries():
            bolt = entry.getFromNodePath().getPythonTag("bolt")
            point = entry.getSurfacePoint(self.render)
            dist = (point - bolt.prev).lengthSquared()
            if bolt not in nearest or dist < nearest[bolt][1]:
                nearest[bolt] = (entry, dist, point)
        dead = set()
        for bolt, (entry, _, point) in nearest.items():
            into = entry.getIntoNodePath()
            drone = into.getPythonTag("drone")
            if drone and drone in self.drones:
                if drone.take_hit(bolt.vel):
                    self.kill_drone(drone, point)
                    point = None
            elif into.getPythonTag("player"):
                self.damage_player(ENEMY_DAMAGE, bolt.vel)
                point = None
            if point is not None:
                color = (1.8, 0.4, 0.2) if bolt.enemy else (1.2, 0.8, 0.3)
                self.flashes.append(Flash(self, point, color, 0.25, 0.25))
            dead.add(bolt)
        for b in self.bolts:
            if b.age > BOLT_LIFE:
                dead.add(b)
        for b in dead:
            b.destroy()
        self.bolts = [b for b in self.bolts if b not in dead]


def screenshot_test(game):
    """Headless smoke test: fire a few bolts, render, save a screenshot, exit."""
    game.player.setH(20)

    def shoot(task):
        game.fire()
        return task.done

    def snap(task):
        game.win.saveScreenshot(sys.argv[sys.argv.index("--screenshot") + 1])
        sys.exit(0)

    game.taskMgr.doMethodLater(0.5, shoot, "shoot")
    game.taskMgr.doMethodLater(0.55, snap, "snap")


if __name__ == "__main__":
    game = Game()
    if SCREENSHOT_MODE:
        screenshot_test(game)
    game.run()
