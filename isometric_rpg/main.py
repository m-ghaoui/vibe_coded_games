"""Isometric field with a warrior fighting skeletons. All geometry is generated in code."""

import argparse
import math
import random
import sys

from panda3d.core import (
    AmbientLight,
    AntialiasAttrib,
    CullFaceAttrib,
    DirectionalLight,
    Filename,
    KeyboardButton,
    MouseButton,
    OrthographicLens,
    Point3,
    RenderState,
    TextNode,
    Vec3,
    WindowProperties,
    loadPrcFileData,
)

from field import FIELD_SIZE, build_field, is_free, resolve_collision
from skeleton import Skeleton, forward_of
from warrior import ATTACK_HIT_AT, ATTACK_TIME, RUN_SPEED, TURN_RATE, WARRIOR_RADIUS, Warrior

MAX_HEALTH = 100
REGEN_DELAY = 4.0      # seconds without damage before health regenerates
REGEN_RATE = 6.0       # health per second
INVULNERABLE_TIME = 0.5
SWORD_REACH = 1.6
SWORD_ARC = 75.0       # degrees either side of facing
DEATH_TIME = 0.8
SPAWN_INTERVAL = 3.0
SPAWN_MIN_DIST = 6.0


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", help="render offscreen and save a screenshot to this path")
    parser.add_argument("--frames", type=int, default=30, help="frame at which to take the screenshot")
    parser.add_argument("--time", type=float, default=0.0, help="animation time offset (seconds)")
    parser.add_argument("--zoom", type=float, default=7.0, help="visible height in world units")
    parser.add_argument("--keys", default="", help="movement keys held down, e.g. 'wd' (for screenshots)")
    parser.add_argument("--attack", action="store_true", help="swing the sword continuously (for screenshots)")
    parser.add_argument("--spawn-near", action="store_true", help="start with skeletons close by")
    args = parser.parse_args()

    loadPrcFileData("", "win-size 1280 720\nframebuffer-multisample 1\nmultisamples 4\n")
    if args.screenshot:
        loadPrcFileData("", "window-type offscreen\n")

    from direct.gui.DirectGui import DirectFrame
    from direct.gui.OnscreenText import OnscreenText
    from direct.showbase.ShowBase import ShowBase

    class Game(ShowBase):
        def __init__(self):
            super().__init__()
            self.disableMouse()
            self.setBackgroundColor(0.55, 0.72, 0.88, 1)
            if self.win and not args.screenshot:
                props = WindowProperties()
                props.setTitle("Isometric RPG")
                self.win.requestProperties(props)

            self.rng = random.Random(7)
            field, self.obstacles = build_field(self.rng)
            field.reparentTo(self.render)
            self.warrior = Warrior(self.render)
            self.warrior.root.setH(25)

            self.setup_lights()
            self.render.setShaderAuto()
            self.render.setAntialias(AntialiasAttrib.MAuto)

            self.zoom = args.zoom
            self.target = Point3(0, 0, 0.9)
            self.cam_offset = Vec3(40, -40, 40)  # isometric diagonal: 45 deg heading, 35.26 deg pitch
            self.lens = OrthographicLens()
            self.cam.node().setLens(self.lens)
            self.lens.setNearFar(1, 200)
            self.camera.setPos(self.target + self.cam_offset)
            self.camera.lookAt(self.target)
            self.update_lens()

            self.accept("window-event", lambda w: self.update_lens())
            self.accept("wheel_up", self.set_zoom, [0.9])
            self.accept("wheel_down", self.set_zoom, [1 / 0.9])
            self.accept("mouse1", self.start_attack)
            self.accept("r", self.rise_again)
            self.accept("escape", sys.exit)

            # Screen-relative directions on the ground: up the screen is away from the camera.
            self.screen_up = Vec3(-1, 1, 0).normalized()
            self.screen_right = Vec3(1, 1, 0).normalized()
            self.move_keys = {
                "w": (KeyboardButton.asciiKey("w"), KeyboardButton.up()),
                "s": (KeyboardButton.asciiKey("s"), KeyboardButton.down()),
                "a": (KeyboardButton.asciiKey("a"), KeyboardButton.left()),
                "d": (KeyboardButton.asciiKey("d"), KeyboardButton.right()),
            }

            self.setup_hud()
            self.skeletons = []
            self.reset_state()
            if args.spawn_near:
                for ang in (30, 150, 270):
                    r = math.radians(ang)
                    self.spawn_skeleton(Point3(3 * math.cos(r), 3 * math.sin(r), 0))

            self.frames = 0
            self.taskMgr.add(self.update, "update")

        def reset_state(self):
            self.velocity = Vec3(0, 0, 0)
            self.health = MAX_HEALTH
            self.since_hurt = REGEN_DELAY
            self.hurt_flash = 0.0
            self.attack_t = None      # seconds into the current swing
            self.attack_hit = set()   # skeletons already hit by this swing
            self.dead_t = None
            self.kills = 0
            self.spawn_timer = 1.0
            self.death_text.hide()

        def rise_again(self):
            if self.dead_t is None:
                return
            for sk in self.skeletons:
                if sk.alive:
                    sk.root.removeNode()
                for node, _, _ in sk.pieces:
                    node.removeNode()
            self.skeletons = []
            self.warrior.root.setPos(0, 0, 0)
            self.warrior.root.setH(25)
            self.warrior.body.clearColorScale()
            self.reset_state()

        # -- setup -----------------------------------------------------------

        def setup_lights(self):
            amb = AmbientLight("ambient")
            amb.setColor((0.38, 0.42, 0.52, 1))
            self.render.setLight(self.render.attachNewNode(amb))

            sun = DirectionalLight("sun")
            sun.setColor((1.05, 0.97, 0.85, 1))
            sun.setShadowCaster(True, 2048, 2048)
            # Render back faces into the shadow map to avoid self-shadowing acne.
            sun.setInitialState(RenderState.make(CullFaceAttrib.makeReverse()))
            lens = sun.getLens()
            lens.setFilmSize(30, 30)
            lens.setNearFar(5, 80)
            sun_np = self.render.attachNewNode(sun)
            sun_np.setPos(-14, -20, 30)
            sun_np.lookAt(0, 0, 0)
            self.render.setLight(sun_np)

            # The two-sided cloak can't avoid self-shadowing acne, so it is lit by an
            # identical sun that ignores the shadow map. It still casts a shadow itself.
            flat = DirectionalLight("sun_no_shadow")
            flat.setColor(sun.getColor())
            flat_np = self.render.attachNewNode(flat)
            flat_np.setHpr(sun_np.getHpr())
            cloak = self.warrior.cloak.np
            cloak.setLightOff(sun_np)
            cloak.setLight(flat_np)

        def setup_hud(self):
            DirectFrame(parent=self.a2dTopLeft, frameColor=(0, 0, 0, 0.55),
                        frameSize=(0, 0.66, -0.07, 0), pos=(0.05, 0, -0.05))
            self.health_fill = DirectFrame(parent=self.a2dTopLeft, frameColor=(0.80, 0.12, 0.10, 1),
                                           frameSize=(0, 0.62, -0.05, 0), pos=(0.07, 0, -0.06))
            self.kills_text = OnscreenText(parent=self.a2dTopRight, text="", pos=(-0.06, -0.11),
                                           scale=0.06, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 0.8),
                                           align=TextNode.ARight, mayChange=True)
            OnscreenText(parent=self.a2dBottomCenter, pos=(0, 0.06), scale=0.045,
                         text="WASD: move    Left click (hold): attack    Wheel: zoom    Esc: quit",
                         fg=(1, 1, 1, 0.85), shadow=(0, 0, 0, 0.8))
            self.death_text = OnscreenText(text="You have fallen.\nPress R to rise again.",
                                           pos=(0, 0.1), scale=0.09, fg=(0.95, 0.85, 0.75, 1),
                                           shadow=(0, 0, 0, 1))

        def update_lens(self):
            aspect = self.getAspectRatio() if self.win else 16 / 9
            self.lens.setFilmSize(self.zoom * aspect, self.zoom)

        def set_zoom(self, factor):
            self.zoom = max(3.0, min(26.0, self.zoom * factor))
            self.update_lens()

        # -- input -----------------------------------------------------------

        def held(self, key):
            if key in args.keys:
                return True
            mw = self.mouseWatcherNode
            return mw is not None and any(mw.isButtonDown(b) for b in self.move_keys[key])

        def mouse_ground_point(self):
            """Where the mouse cursor points on the ground plane (z=0), or None."""
            mw = self.mouseWatcherNode
            if mw is None or not mw.hasMouse():
                return None
            near, far = Point3(), Point3()
            if not self.lens.extrude(mw.getMouse(), near, far):
                return None
            near = self.render.getRelativePoint(self.cam, near)
            far = self.render.getRelativePoint(self.cam, far)
            if abs(near.z - far.z) < 1e-6:
                return None
            return near + (far - near) * (near.z / (near.z - far.z))

        def start_attack(self):
            if self.dead_t is not None or self.attack_t is not None:
                return
            self.attack_t = 0.0
            self.attack_hit = set()
            # Turn to face the cursor.
            point = self.mouse_ground_point()
            root = self.warrior.root
            if point is not None:
                d = point - root.getPos()
                if d.x * d.x + d.y * d.y > 1e-4:
                    root.setH(math.degrees(math.atan2(d.x, -d.y)))

        # -- combat ----------------------------------------------------------

        def sword_hits(self):
            root = self.warrior.root
            pos, fwd = root.getPos(), forward_of(root.getH())
            for sk in self.skeletons:
                if not sk.hittable or sk in self.attack_hit:
                    continue
                to = sk.position() - pos
                to.z = 0
                dist = to.length()
                if dist > SWORD_REACH + 0.3:
                    continue
                if dist > 1e-6 and fwd.dot(to / dist) < math.cos(math.radians(SWORD_ARC)):
                    continue
                self.attack_hit.add(sk)
                sk.hurt(to if dist > 1e-6 else fwd)
                if not sk.alive:
                    self.kills += 1

        def warrior_hit(self, skeleton, damage):
            if self.dead_t is not None or self.since_hurt < INVULNERABLE_TIME:
                return
            self.health = max(0, self.health - damage)
            self.since_hurt = 0.0
            self.hurt_flash = 1.0
            push = self.warrior.root.getPos() - skeleton.position()
            push.z = 0
            if push.lengthSquared() > 1e-6:
                self.velocity += push.normalized() * 5.0
            if self.health <= 0:
                self.dead_t = 0.0
                self.attack_t = None

        def spawn_skeleton(self, pos=None):
            if pos is None:
                wpos = self.warrior.root.getPos()
                half = FIELD_SIZE / 2 - 1
                for _ in range(50):
                    x, y = self.rng.uniform(-half, half), self.rng.uniform(-half, half)
                    if math.hypot(x - wpos.x, y - wpos.y) > SPAWN_MIN_DIST and is_free(x, y, 0.6, self.obstacles):
                        pos = Point3(x, y, 0)
                        break
                else:
                    return
            self.skeletons.append(Skeleton(self.render, pos, self.rng))

        def separate(self):
            """Push skeletons apart from each other and out of the warrior."""
            alive = [sk for sk in self.skeletons if sk.hittable]
            wpos = self.warrior.root.getPos()
            for i, a in enumerate(alive):
                pa = a.position()
                for b in alive[i + 1:]:
                    pb = b.position()
                    d = pa - pb
                    d.z = 0
                    dist = d.length()
                    if 1e-6 < dist < 2 * 0.3:
                        push = d / dist * (0.6 - dist) * 0.5
                        pa += push
                        b.root.setPos(resolve_collision(pb - push, 0.3, self.obstacles))
                d = pa - wpos
                d.z = 0
                dist = d.length()
                if 1e-6 < dist < 0.3 + WARRIOR_RADIUS:
                    pa += d / dist * (0.3 + WARRIOR_RADIUS - dist)
                a.root.setPos(resolve_collision(pa, 0.3, self.obstacles))

        # -- frame -----------------------------------------------------------

        def update(self, task):
            dt = min(globalClock.getDt(), 0.1)
            if args.screenshot:
                dt = 1 / 60  # offscreen frames run unthrottled; keep motion deterministic
            t = globalClock.getFrameTime() + args.time
            if args.screenshot:
                t = self.frames / 60 + args.time
            alive = self.dead_t is None

            # Holding the mouse button keeps swinging, re-aiming at the cursor each time.
            mw = self.mouseWatcherNode
            mouse_held = mw is not None and mw.isButtonDown(MouseButton.one())
            if (args.attack or mouse_held) and alive:
                self.start_attack()

            move = Vec3(0, 0, 0)
            if alive:
                if self.held("w"):
                    move += self.screen_up
                if self.held("s"):
                    move -= self.screen_up
                if self.held("d"):
                    move += self.screen_right
                if self.held("a"):
                    move -= self.screen_right
                if move.lengthSquared() > 0:
                    move.normalize()

            # Accelerate quickly toward the desired velocity, stop a bit softer.
            # Swinging the sword slows you down.
            top_speed = RUN_SPEED * (0.5 if self.attack_t is not None else 1.0)
            rate = 12 if move.lengthSquared() > 0 else 9
            self.velocity += (move * top_speed - self.velocity) * min(1.0, rate * dt)
            root = self.warrior.root
            pos = root.getPos() + self.velocity * dt
            pos = resolve_collision(pos, WARRIOR_RADIUS, self.obstacles)
            moved = (pos - root.getPos()) / dt if dt > 0 else Vec3(0, 0, 0)
            root.setPos(pos)

            # Turn to face the movement direction (the model faces -Y at heading 0),
            # unless mid-swing: then keep facing the attack direction.
            if move.lengthSquared() > 0 and self.attack_t is None:
                want = math.degrees(math.atan2(move.x, -move.y))
                diff = (want - root.getH() + 180) % 360 - 180
                step = TURN_RATE * dt
                root.setH(root.getH() + max(-step, min(step, diff)))

            # Sword swing.
            attack_p = None
            if self.attack_t is not None:
                prev = self.attack_t / ATTACK_TIME
                self.attack_t += dt
                attack_p = min(1.0, self.attack_t / ATTACK_TIME)
                if prev < ATTACK_HIT_AT <= attack_p:
                    self.sword_hits()
                if attack_p >= 1.0:
                    self.attack_t = None

            # Health: regen, death.
            self.since_hurt += dt
            if alive and self.since_hurt > REGEN_DELAY:
                self.health = min(MAX_HEALTH, self.health + REGEN_RATE * dt)
            dead_p = 0.0
            if self.dead_t is not None:
                self.dead_t += dt
                dead_p = min(1.0, self.dead_t / DEATH_TIME)
                if dead_p >= 1.0:
                    self.death_text.show()

            self.warrior.animate(t, dt, moved.length(), attack_p, dead_p)

            # Red hit flash.
            if self.hurt_flash > 0:
                self.hurt_flash = max(0.0, self.hurt_flash - dt * 4)
                f = self.hurt_flash
                self.warrior.body.setColorScale(1 + 0.8 * f, 1 - 0.5 * f, 1 - 0.5 * f, 1)
            else:
                self.warrior.body.clearColorScale()

            # Skeletons.
            for sk in self.skeletons:
                if sk.update(dt, t, pos, alive, self.obstacles):
                    self.warrior_hit(sk, 12)
            self.separate()
            self.skeletons = [sk for sk in self.skeletons if not sk.removed]

            # Spawning: more skeletons as the kill count grows.
            living = sum(1 for sk in self.skeletons if sk.alive)
            wanted = min(8, 4 + self.kills // 3)
            self.spawn_timer -= dt
            if alive and living < wanted and self.spawn_timer <= 0:
                self.spawn_skeleton()
                self.spawn_timer = SPAWN_INTERVAL

            # HUD.
            self.health_fill.setScale(max(0.001, self.health / MAX_HEALTH), 1, 1)
            self.kills_text.setText(f"Skeletons slain: {self.kills}")

            # Camera follows, keeping the isometric angle.
            self.target += (pos + Vec3(0, 0, 0.9) - self.target) * min(1.0, 6 * dt)
            self.camera.setPos(self.target + self.cam_offset)

            self.frames += 1
            if args.screenshot and self.frames == args.frames:
                self.win.saveScreenshot(Filename.fromOsSpecific(args.screenshot))
                sys.exit(0)
            return task.cont

    Game().run()


if __name__ == "__main__":
    main()
