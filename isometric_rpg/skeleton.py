"""Skeleton enemies: model, animation, simple AI and falling apart when killed."""

import math

from panda3d.core import Point3, Vec3

from field import FIELD_SIZE
from geometry import MeshBuilder, add_box, box_node
from warrior import lerp, sample_keys, smoothstep

BONE = (0.88, 0.85, 0.74, 1)
BONE_DARK = (0.66, 0.62, 0.52, 1)
SOCKET = (0.08, 0.06, 0.06, 1)
EYE_GLOW = (1.0, 0.25, 0.12, 1)
IRON = (0.30, 0.30, 0.32, 1)
RUST = (0.56, 0.36, 0.22, 1)
GRIP = (0.25, 0.17, 0.10, 1)

RADIUS = 0.3
HEALTH = 3
WANDER_SPEED = 1.2
CHASE_SPEED = 2.4
AGGRO_RANGE = 7.0
ATTACK_RANGE = 1.2   # start winding up within this distance
HIT_RANGE = 1.4      # the strike lands if the warrior is still this close...
HIT_ANGLE = 60.0     # ...and within this many degrees of where the skeleton faces
DAMAGE = 12
TURN_RATE = 360.0

RISE_TIME = 1.2
WINDUP_TIME = 0.5
STRIKE_TIME = 0.15
RECOVER_TIME = 0.6
HURT_TIME = 0.25

ARM_R_READY = -30.0
ELBOW_R_READY = -50.0
SWORD_READY = 125.0  # blade forward and up

# Attack keyframes over windup + strike + recover: (progress, arm_r P, elbow P, sword P, torso P).
ATTACK_KEYS = [
    (0.00, ARM_R_READY, ELBOW_R_READY, SWORD_READY, 8),
    (0.40, -165, -30, 180, -6),   # sword raised high (end of windup)
    (0.50, -55, -10, 180, 18),    # chop (strike)
    (0.75, -55, -10, 180, 18),
    (1.00, ARM_R_READY, ELBOW_R_READY, SWORD_READY, 8),
]
ATTACK_TOTAL = WINDUP_TIME + STRIKE_TIME + RECOVER_TIME


def heading_to(dx, dy):
    """Heading that makes a model facing -Y look along (dx, dy)."""
    return math.degrees(math.atan2(dx, -dy))


def forward_of(h):
    r = math.radians(h)
    return Vec3(math.sin(r), -math.cos(r), 0)


class Skeleton:
    def __init__(self, parent, pos, rng):
        self.rng = rng
        self.root = parent.attachNewNode("skeleton")
        self.root.setPos(pos)
        self.root.setH(rng.uniform(0, 360))
        self.body = self.root.attachNewNode("body")
        self.build()

        self.health = HEALTH
        self.state = "rise"
        self.timer = 0.0
        self.phase = rng.uniform(0, 2 * math.pi)
        self.walk = 0.0
        self.speed = 0.0
        self.knock = Vec3(0, 0, 0)
        self.wander_target = None
        self.pause = 0.0
        self.flash = 0.0
        self.attack_p = None
        self.pieces = []  # (node, velocity, spin) once dead
        self.removed = False

    # -- model ---------------------------------------------------------------

    def build(self):
        self.hips = self.body.attachNewNode("hips")
        self.hips.setZ(0.88)
        box_node("pelvis", -0.14, 0.14, -0.07, 0.07, -0.08, 0.03, BONE).reparentTo(self.hips)

        self.legs = []
        for side in (-1, 1):
            leg = self.hips.attachNewNode("leg")
            leg.setPos(0.08 * side, 0, -0.05)
            b = MeshBuilder("leg")
            add_box(b, -0.03, 0.03, -0.03, 0.03, -0.4, 0.0, BONE)          # femur
            add_box(b, -0.045, 0.045, -0.045, 0.045, -0.45, -0.37, BONE_DARK)  # knee
            add_box(b, -0.026, 0.026, -0.026, 0.026, -0.79, -0.43, BONE)   # tibia
            add_box(b, -0.045, 0.045, -0.13, 0.04, -0.83, -0.77, BONE)     # foot
            b.node().reparentTo(leg)
            self.legs.append(leg)

        self.torso = self.hips.attachNewNode("torso")
        self.torso.setZ(0.03)
        b = MeshBuilder("ribcage")
        add_box(b, -0.03, 0.03, 0.03, 0.09, 0.0, 0.5, BONE_DARK)           # spine
        for i in range(4):
            z = 0.17 + i * 0.075
            w = 0.12 + 0.012 * i
            add_box(b, -w, w, -0.09, 0.08, z, z + 0.035, BONE)             # ribs
        add_box(b, -0.025, 0.025, -0.105, -0.085, 0.17, 0.42, BONE)        # sternum
        add_box(b, -0.25, 0.25, -0.03, 0.03, 0.43, 0.47, BONE)             # collarbones
        add_box(b, -0.025, 0.025, -0.025, 0.025, 0.47, 0.53, BONE_DARK)    # neck
        b.node().reparentTo(self.torso)

        self.head = self.torso.attachNewNode("head")
        self.head.setZ(0.52)
        b = MeshBuilder("skull")
        add_box(b, -0.1, 0.1, -0.11, 0.1, 0.04, 0.24, BONE)                # cranium
        add_box(b, -0.075, 0.075, -0.1, 0.04, -0.02, 0.05, BONE_DARK)      # jaw
        add_box(b, -0.065, 0.065, -0.112, -0.1, 0.035, 0.05, (0.95, 0.93, 0.85, 1))  # teeth
        for side in (-1, 1):
            x = 0.045 * side
            add_box(b, x - 0.032, x + 0.032, -0.116, -0.1, 0.1, 0.16, SOCKET)
        add_box(b, -0.014, 0.014, -0.116, -0.1, 0.065, 0.095, SOCKET)      # nose hole
        b.node().reparentTo(self.head)
        e = MeshBuilder("eyes")
        for side in (-1, 1):
            x = 0.045 * side
            add_box(e, x - 0.014, x + 0.014, -0.12, -0.112, 0.118, 0.142, EYE_GLOW)
        self.eyes = e.node()
        self.eyes.reparentTo(self.head)
        self.eyes.setLightOff(1)  # unlit, so they glow

        self.arm_l = self.torso.attachNewNode("arm_l")
        self.arm_l.setPos(-0.24, 0, 0.44)
        b = MeshBuilder("arm_l")
        add_box(b, -0.025, 0.025, -0.025, 0.025, -0.3, 0.0, BONE)
        add_box(b, -0.035, 0.035, -0.035, 0.035, -0.33, -0.27, BONE_DARK)
        add_box(b, -0.022, 0.022, -0.022, 0.022, -0.58, -0.3, BONE)
        add_box(b, -0.035, 0.035, -0.03, 0.03, -0.66, -0.58, BONE)
        b.node().reparentTo(self.arm_l)

        self.arm_r = self.torso.attachNewNode("arm_r")
        self.arm_r.setPos(0.24, 0, 0.44)
        b = MeshBuilder("upper_r")
        add_box(b, -0.025, 0.025, -0.025, 0.025, -0.3, 0.0, BONE)
        add_box(b, -0.035, 0.035, -0.035, 0.035, -0.33, -0.27, BONE_DARK)
        b.node().reparentTo(self.arm_r)
        self.elbow_r = self.arm_r.attachNewNode("elbow_r")
        self.elbow_r.setZ(-0.3)
        b = MeshBuilder("forearm_r")
        add_box(b, -0.022, 0.022, -0.022, 0.022, -0.27, 0.0, BONE)
        add_box(b, -0.035, 0.035, -0.03, 0.03, -0.35, -0.27, BONE)
        b.node().reparentTo(self.elbow_r)

        self.sword = self.elbow_r.attachNewNode("sword")
        self.sword.setZ(-0.31)
        b = MeshBuilder("rusty_sword")
        add_box(b, -0.018, 0.018, -0.018, 0.018, -0.07, 0.07, GRIP)
        add_box(b, -0.03, 0.03, -0.03, 0.03, -0.1, -0.07, IRON)
        add_box(b, -0.1, 0.1, -0.02, 0.02, 0.07, 0.1, IRON)
        add_box(b, -0.028, 0.028, -0.008, 0.008, 0.1, 0.62, RUST)
        b.tri((-0.028, 0, 0.62), (0.028, 0, 0.62), (0, 0, 0.7), RUST, (0, -1, 0))
        b.tri((-0.028, 0, 0.62), (0.028, 0, 0.62), (0, 0, 0.7), RUST, (0, 1, 0))
        b.node().reparentTo(self.sword)

    # -- state ---------------------------------------------------------------

    @property
    def alive(self):
        return self.state not in ("dead",)

    @property
    def hittable(self):
        return self.state not in ("dead", "rise")

    def position(self):
        return self.root.getPos()

    def hurt(self, push):
        """Take one hit; push is a horizontal knockback direction."""
        if not self.hittable:
            return
        self.health -= 1
        self.flash = 1.0
        self.knock = Vec3(push.x, push.y, 0).normalized() * 4.0
        if self.health <= 0:
            self.die(push)
        else:
            self.state, self.timer, self.attack_p = "hurt", 0.0, None

    def die(self, push):
        self.state, self.timer = "dead", 0.0
        self.eyes.removeNode()
        self.body.clearColorScale()
        render = self.root.getTop()
        center = self.root.getPos() + Vec3(0, 0, 0.6)
        for gn in self.root.findAllMatches("**/+GeomNode"):
            # Re-pivot each bone on its own centre so it tumbles naturally.
            lo, hi = gn.getTightBounds(render)
            mid = (lo + hi) * 0.5
            holder = render.attachNewNode("bone")
            holder.setPos(mid)
            gn.wrtReparentTo(holder)
            out = mid - center
            out.z = 0
            vel = out * 2.5 + Vec3(push.x, push.y, 0) * 1.5 + Vec3(
                self.rng.uniform(-0.6, 0.6), self.rng.uniform(-0.6, 0.6), self.rng.uniform(1.5, 3.0))
            spin = Vec3(*(self.rng.uniform(-400, 400) for _ in range(3)))
            self.pieces.append([holder, vel, spin])
        self.root.removeNode()

    # -- update --------------------------------------------------------------

    def update(self, dt, t, target, target_alive, obstacles):
        """Advance AI and animation. Returns True on the frame a strike lands."""
        if self.state == "dead":
            self.update_pieces(dt)
            return False

        self.timer += dt
        pos = self.root.getPos()
        to = Vec3(target.x - pos.x, target.y - pos.y, 0)
        dist = to.length()
        move = Vec3(0, 0, 0)
        speed = 0.0
        struck = False

        if self.state == "rise":
            f = min(1.0, self.timer / RISE_TIME)
            self.body.setZ(-0.95 * (1 - smoothstep(f)))
            self.body.setR(8 * math.sin(self.timer * 18) * (1 - f))
            if f >= 1:
                self.body.setZ(0)
                self.body.setR(0)
                self.state, self.timer = "wander", 0.0
        elif self.state in ("wander", "chase"):
            if target_alive and dist < AGGRO_RANGE:
                self.state = "chase"
            elif self.state == "chase":
                self.state, self.wander_target = "wander", None
            if self.state == "chase":
                if dist < ATTACK_RANGE:
                    self.state, self.timer = "windup", 0.0
                else:
                    move, speed = to.normalized(), CHASE_SPEED
            else:
                if self.pause > 0:
                    self.pause -= dt
                elif self.wander_target is None:
                    half = FIELD_SIZE / 2 - 1
                    self.wander_target = Point3(
                        max(-half, min(half, pos.x + self.rng.uniform(-4, 4))),
                        max(-half, min(half, pos.y + self.rng.uniform(-4, 4))), 0)
                else:
                    d = self.wander_target - pos
                    d.z = 0
                    if d.length() < 0.3 or self.timer > 6:
                        self.wander_target, self.timer = None, 0.0
                        self.pause = self.rng.uniform(0.5, 2.5)
                    else:
                        move, speed = d.normalized(), WANDER_SPEED
        elif self.state == "windup":
            # Keep tracking the target while raising the sword.
            move = to.normalized() if dist > 0 else Vec3(0, 0, 0)
            if self.timer >= WINDUP_TIME:
                self.state, self.timer = "strike", 0.0
        elif self.state == "strike":
            if self.timer >= STRIKE_TIME:
                self.state, self.timer = "recover", 0.0
                if target_alive and dist < HIT_RANGE:
                    fwd = forward_of(self.root.getH())
                    cos = fwd.dot(to.normalized()) if dist > 1e-6 else 1
                    struck = cos > math.cos(math.radians(HIT_ANGLE))
        elif self.state == "recover":
            if self.timer >= RECOVER_TIME:
                self.state, self.timer = "chase", 0.0
        elif self.state == "hurt":
            if self.timer >= HURT_TIME:
                self.state, self.timer = "chase", 0.0

        # Face the way we're going (or the target while attacking).
        if move.lengthSquared() > 0:
            want = heading_to(move.x, move.y)
            diff = (want - self.root.getH() + 180) % 360 - 180
            step = TURN_RATE * dt
            self.root.setH(self.root.getH() + max(-step, min(step, diff)))

        # Only walk forward once roughly facing the direction of travel.
        if speed > 0:
            facing = forward_of(self.root.getH())
            speed *= max(0.0, facing.dot(move))
        new_pos = pos + move * speed * dt + self.knock * dt
        self.knock *= max(0.0, 1 - 8 * dt)
        self.root.setPos(new_pos)
        self.speed = speed

        # Attack progress across windup/strike/recover.
        if self.state == "windup":
            self.attack_p = self.timer / ATTACK_TOTAL
        elif self.state == "strike":
            self.attack_p = (WINDUP_TIME + self.timer) / ATTACK_TOTAL
        elif self.state == "recover":
            self.attack_p = (WINDUP_TIME + STRIKE_TIME + self.timer) / ATTACK_TOTAL
        else:
            self.attack_p = None

        self.animate(t, dt)
        return struck

    def animate(self, t, dt):
        target = min(1.0, self.speed / CHASE_SPEED * 1.4)
        self.walk += (target - self.walk) * min(1.0, dt * 8)
        walk = self.walk
        # A slightly shorter, stiffer stride than the warrior.
        self.phase = (self.phase + dt * self.speed * 2 * math.pi / 1.3) % (2 * math.pi)
        stride = math.sin(self.phase)
        bob = abs(math.cos(self.phase))

        self.hips.setZ(0.88 + walk * (0.035 * bob - 0.03))
        self.hips.setR(walk * 4 * stride)  # lurching side to side
        for leg, side in zip(self.legs, (-1, 1)):
            leg.setP(walk * 32 * stride * side)

        # Hunched, with an idle rattle.
        rattle = math.sin(t * 2.3 + self.phase)
        self.torso.setP(8 + 2 * rattle * (1 - walk))
        self.torso.setR(-walk * 3 * stride)
        self.head.setH(10 * math.sin(t * 0.8 + self.phase) * (1 - walk) + 4 * math.sin(t * 7.0) * walk)
        self.head.setR(6 * math.sin(t * 1.1 + 2))

        self.arm_l.setP(-walk * 35 * stride + 3 * rattle)
        self.arm_l.setR(6)
        self.arm_r.setP(ARM_R_READY + walk * 12 * stride)
        self.arm_r.setR(-6)
        self.elbow_r.setP(ELBOW_R_READY)
        self.sword.setP(SWORD_READY)

        if self.attack_p is not None:
            arm_p, elbow_p, sword_p, torso_p = sample_keys(ATTACK_KEYS, self.attack_p)
            self.arm_r.setP(arm_p)
            self.elbow_r.setP(elbow_p)
            self.sword.setP(sword_p)
            self.torso.setP(torso_p)

        if self.state == "hurt":
            f = 1 - self.timer / HURT_TIME
            self.torso.setP(-18 * f)
            self.head.setP(-15 * f)
        else:
            self.head.setP(0)

        # Hit flash: bright white, fading out.
        if self.flash > 0:
            self.flash = max(0.0, self.flash - dt * 5)
            k = 1 + 1.5 * self.flash
            self.body.setColorScale(k, k, k, 1)
        else:
            self.body.clearColorScale()

    def update_pieces(self, dt):
        self.timer += dt
        for piece in self.pieces:
            node, vel, spin = piece
            pos = node.getPos()
            if pos.z > 0.05 or vel.z > 0:
                vel.z -= 9.8 * dt
                node.setPos(pos + vel * dt)
                node.setHpr(node.getHpr() + spin * dt)
                if node.getZ() <= 0.05:
                    node.setZ(0.05)
                    vel.x *= 0.3
                    vel.y *= 0.3
                    vel.z = -vel.z * 0.25 if vel.z < -1.5 else 0
                    spin *= 0.3
            else:
                # Settle: slide to a stop and roll flat onto the ground.
                node.setPos(pos + Vec3(vel.x, vel.y, 0) * dt)
                vel *= max(0.0, 1 - 6 * dt)
                p, r = node.getP(), node.getR()
                node.setP(lerp(p, round(p / 90) * 90, min(1.0, 6 * dt)))
                node.setR(lerp(r, round(r / 90) * 90, min(1.0, 6 * dt)))
            if self.timer > 4.0:
                node.setZ(node.getZ() - 0.25 * dt)  # sink into the earth
        if self.timer > 6.0:
            for node, _, _ in self.pieces:
                node.removeNode()
            self.pieces = []
            self.removed = True
