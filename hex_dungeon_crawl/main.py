"""Hex Dungeon Crawl: a step-and-turn first-person dungeon crawler on a hex grid.

Instead of four directions, every chamber is a hexagon with six sides. You
turn in 60 degree steps and can step to any of the six neighbouring cells
without turning. Find the glowing stairs to go deeper.

Everything (maze, geometry, textures) is generated at startup.
Run with:  .venv/bin/python main.py
"""

import math
import random
import sys

from panda3d.core import (
    loadPrcFileData,
    AmbientLight,
    CardMaker,
    CullBinManager,
    Filename,
    Fog,
    KeyboardButton,
    LineSegs,
    PointLight,
    TextNode,
    TransparencyAttrib,
)

import hexgrid as hg
from dungeon_mesh import CELL_SIZE, LevelTextures, build_level
from minimap import Minimap

SCREENSHOT_MODE = "--screenshot" in sys.argv

loadPrcFileData("", """
window-title Panda3D hex dungeon crawl
win-size 1280 720
sync-video true
show-frame-rate-meter true
framebuffer-multisample 1
multisamples 4
""")
if SCREENSHOT_MODE:
    loadPrcFileData("", "window-type offscreen\nsync-video false\nshow-frame-rate-meter false")

from direct.showbase.ShowBase import ShowBase  # noqa: E402  (after prc config)
from direct.gui.OnscreenText import OnscreenText  # noqa: E402

EYE_HEIGHT = 1.55
CAMERA_PITCH = -6.0
MOVE_TIME = 0.28
TURN_TIME = 0.20
BUMP_TIME = 0.22
FADE_TIME = 0.5

# Step keys, relative to the current heading.
STEPS = {"forward": 0, "forward_right": 1, "back_right": 2, "back": 3, "back_left": 4, "forward_left": 5}
TURNS = {"turn_left": -1, "turn_right": 1}
KEYS = {
    "forward": ["w", "arrow_up"],
    "back": ["s", "arrow_down"],
    "turn_left": ["a", "arrow_left"],
    "turn_right": ["d", "arrow_right"],
    "forward_left": ["q"],
    "forward_right": ["e"],
    "back_left": ["z"],
    "back_right": ["c"],
}
SPECIAL_BUTTONS = {
    "arrow_up": KeyboardButton.up(),
    "arrow_down": KeyboardButton.down(),
    "arrow_left": KeyboardButton.left(),
    "arrow_right": KeyboardButton.right(),
}

# Each depth gets its own stone tint, fog and torch colour (cycles after the last).
DEPTH_STYLES = [
    dict(name="The Grey Halls", tint=(1.0, 1.0, 1.0, 1), fog=(0.020, 0.018, 0.016), torch=(1.0, 0.70, 0.42)),
    dict(name="The Mossy Deep", tint=(0.80, 1.0, 0.75, 1), fog=(0.012, 0.022, 0.014), torch=(1.0, 0.78, 0.45)),
    dict(name="The Rust Vaults", tint=(1.10, 0.75, 0.62, 1), fog=(0.025, 0.012, 0.010), torch=(1.0, 0.62, 0.35)),
    dict(name="The Frozen Cells", tint=(0.75, 0.88, 1.15, 1), fog=(0.012, 0.016, 0.026), torch=(0.85, 0.80, 0.75)),
    dict(name="The Violet Abyss", tint=(0.92, 0.75, 1.10, 1), fog=(0.018, 0.010, 0.024), torch=(1.0, 0.66, 0.50)),
]

HELP_TEXT = """W / S   step forward / back
A / D   turn left / right
Q / E   step forward-left / -right
Z / C   step back-left / -right
Space   descend stairs
M  map    H  help    Esc  quit"""


def level_radius(depth):
    return min(4 + depth, 11)


def smoothstep(t):
    return t * t * (3 - 2 * t)


class Anim:
    def __init__(self, kind, duration, **kw):
        self.kind = kind
        self.duration = duration
        self.t = 0.0
        self.__dict__.update(kw)


class Game(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
        self.camLens.setFov(90)
        self.camLens.setNearFar(0.05, 200)
        self.render.setShaderAuto()
        self.textures = LevelTextures()

        self.ambient = AmbientLight("ambient")
        self.ambient.setColor((0.06, 0.06, 0.075, 1))
        self.render.setLight(self.render.attachNewNode(self.ambient))

        self.torch = PointLight("torch")
        self.torch.setAttenuation((1.0, 0.0, 0.09))
        self.torch_np = self.camera.attachNewNode(self.torch)
        self.torch_np.setPos(0.35, -0.3, -0.2)     # held low and to the right
        self.render.setLight(self.torch_np)

        self.stairs_light = PointLight("stairs-light")
        self.stairs_light.setAttenuation((1.0, 0.0, 0.12))
        self.stairs_light_np = self.render.attachNewNode(self.stairs_light)
        self.render.setLight(self.stairs_light_np)

        self.fog = Fog("fog")
        self.fog.setExpDensity(0.07)
        self.render.setFog(self.fog)

        CullBinManager.getGlobalPtr().addBin("fade", CullBinManager.BT_fixed, 100)
        cm = CardMaker("fade")
        cm.setFrameFullscreenQuad()
        self.fade_card = self.render2d.attachNewNode(cm.generate())
        self.fade_card.setTransparency(TransparencyAttrib.MAlpha)
        self.fade_card.setBin("fade", 0)
        self.fade_card.setColor(0, 0, 0, 0)
        self.fade = None             # None, or [direction, t]

        self.minimap = Minimap(self)
        self._build_hud()

        self.level = None
        self.depth = 0
        self.anim = None
        self.pending = None
        self.time = 0.0
        self.load_level(1, seed=1 if SCREENSHOT_MODE else None)

        for action, keys in KEYS.items():
            for key in keys:
                self.accept(key, self.queue_action, [action])
        self.accept("space", self.descend)
        self.accept("m", self.toggle_map)
        self.accept("h", self.toggle_help)
        self.accept("n", self.new_level)
        self.accept("escape", sys.exit)
        self.taskMgr.add(self.update, "update")

    # -- HUD ------------------------------------------------------------------

    def _build_hud(self):
        self.status = OnscreenText(parent=self.a2dTopLeft, pos=(0.05, -0.09), scale=0.05,
                                   fg=(0.95, 0.85, 0.65, 1), shadow=(0, 0, 0, 1), align=TextNode.ALeft)
        self.help = OnscreenText(text=HELP_TEXT, parent=self.a2dBottomLeft, pos=(0.05, 0.33), scale=0.042,
                                 fg=(0.85, 0.8, 0.7, 0.9), shadow=(0, 0, 0, 1), align=TextNode.ALeft,
                                 mayChange=False)
        self.prompt = OnscreenText(parent=self.aspect2d, pos=(0, -0.45), scale=0.06,
                                   fg=(0.6, 0.9, 1.0, 1), shadow=(0, 0, 0, 1))
        self.banner = OnscreenText(parent=self.aspect2d, pos=(0, 0.55), scale=0.08,
                                   fg=(0.95, 0.85, 0.65, 1), shadow=(0, 0, 0, 1))
        self.banner_time = 0.0

        # Hex compass rose: the outline and labels turn with the view, the
        # pointer at the top stays fixed.
        self.compass = self.a2dBottomCenter.attachNewNode("compass")
        self.compass.setPos(0, 0, 0.24)
        r = 0.12
        ls = LineSegs()
        ls.setThickness(2)
        ls.setColor(0.85, 0.75, 0.55, 0.9)
        for j in range(7):
            a = math.radians(30 + 60 * j)
            (ls.moveTo if j == 0 else ls.drawTo)(r * math.sin(a), 0, r * math.cos(a))
        self.compass_hex = self.compass.attachNewNode(ls.create())
        ls = LineSegs()
        ls.setThickness(2)
        ls.setColor(1.0, 0.5, 0.25, 1)
        ls.moveTo(-0.03, 0, 0.215)
        ls.drawTo(0, 0, 0.18)
        ls.drawTo(0.03, 0, 0.215)
        self.compass.attachNewNode(ls.create())
        self.compass_labels = []
        for name in hg.DIR_NAMES:
            tn = TextNode("dir-" + name)
            tn.setText(name)
            tn.setAlign(TextNode.ACenter)
            tn.setShadow(0.05, 0.05)
            tn.setShadowColor(0, 0, 0, 1)
            self.compass_labels.append(self.compass.attachNewNode(tn))

    def update_compass(self, view_deg):
        self.compass_hex.setR(-view_deg)
        for d, label in enumerate(self.compass_labels):
            a = math.radians(60 * d - view_deg)
            label.setPos(0.155 * math.sin(a), 0, 0.155 * math.cos(a) - 0.015)
            active = d == self.heading
            label.setScale(0.05 if active else 0.038)
            label.node().setTextColor((1.0, 0.6, 0.3, 1) if active else (0.8, 0.75, 0.65, 0.8))

    def update_status(self):
        self.status.setText("Depth %d  -  %s\nSteps %d    Explored %d%%    Facing %s" % (
            self.depth, self.style["name"], self.steps,
            100 * len(self.discovered) // len(self.dungeon.cells), hg.DIR_NAMES[self.heading]))
        on_stairs = self.cell == self.dungeon.stairs
        self.prompt.setText("Stairs lead down.  Press SPACE to descend." if on_stairs else "")

    def toggle_help(self):
        if self.help.isHidden():
            self.help.show()
        else:
            self.help.hide()

    def toggle_map(self):
        self.minimap.toggle()
        self.minimap.rebuild(self.discovered, self.visited)

    # -- levels ---------------------------------------------------------------

    def load_level(self, depth, seed=None):
        if self.level is not None:
            self.level.root.removeNode()
        if seed is None:
            seed = random.randrange(1 << 30)
        self.depth = depth
        self.style = DEPTH_STYLES[(depth - 1) % len(DEPTH_STYLES)]
        self.dungeon = hg.generate(level_radius(depth), seed, depth)
        self.level = build_level(self.dungeon, self.textures, self.style["tint"])
        self.level.root.reparentTo(self.render)

        fog = self.style["fog"]
        self.fog.setColor(*fog)
        self.setBackgroundColor(*fog, 1)
        self.torch_color = self.style["torch"]
        sx, sy = self.level.stairs_pos
        self.stairs_light_np.setPos(sx, sy, 0.6)

        self.cell = self.dungeon.start
        # Start facing the longest open corridor.
        self.heading = max(range(6), key=lambda d: (len(self.dungeon.ray(self.cell, d)), -d))
        x, y = hg.hex_to_world(self.cell, CELL_SIZE)
        self.cam_pos = [x, y]
        self.cam_h = -60.0 * self.heading
        self.bob = 0.0
        self.anim = None
        self.pending = None
        self.steps = 0
        self.discovered = set()
        self.visited = set()
        self.minimap.set_level(self.dungeon)
        self.reveal()
        self.banner.setText("Depth %d\n%s" % (depth, self.style["name"]))
        self.banner_time = 3.0

    def reveal(self):
        self.visited.add(self.cell)
        self.discovered |= hg.visible_cells(self.dungeon, self.cell)
        self.minimap.rebuild(self.discovered, self.visited)
        self.update_status()

    def descend(self):
        if self.cell == self.dungeon.stairs and self.fade is None:
            self.fade = ["out", 0.0]

    def new_level(self):
        if self.fade is None:
            self.load_level(self.depth)

    # -- movement -------------------------------------------------------------

    def queue_action(self, action):
        self.pending = action

    def held_action(self):
        watcher = self.mouseWatcherNode
        if watcher is None:      # offscreen: no keyboard
            return None
        for action, keys in KEYS.items():
            for key in keys:
                button = SPECIAL_BUTTONS.get(key) or KeyboardButton.asciiKey(key)
                if watcher.isButtonDown(button):
                    return action
        return None

    def start_action(self, action):
        if action in TURNS:
            delta = TURNS[action]
            self.heading = (self.heading + delta) % 6
            self.anim = Anim("turn", TURN_TIME, h0=self.cam_h, h1=self.cam_h - 60.0 * delta)
            self.update_status()
            return
        d = (self.heading + STEPS[action]) % 6
        p0 = tuple(self.cam_pos)
        if self.dungeon.is_open(self.cell, d):
            self.cell = hg.neighbor(self.cell, d)
            self.steps += 1
            p1 = hg.hex_to_world(self.cell, CELL_SIZE)
            self.anim = Anim("move", MOVE_TIME, p0=p0, p1=p1)
            self.reveal()
        else:
            self.anim = Anim("bump", BUMP_TIME, p0=p0, dir=hg.heading_vector(d))

    def step_anim(self, dt):
        a = self.anim
        a.t = min(1.0, a.t + dt / a.duration)
        e = smoothstep(a.t)
        if a.kind == "turn":
            self.cam_h = a.h0 + (a.h1 - a.h0) * e
        elif a.kind == "move":
            self.cam_pos = [a.p0[0] + (a.p1[0] - a.p0[0]) * e, a.p0[1] + (a.p1[1] - a.p0[1]) * e]
            self.bob = 0.05 * math.sin(math.pi * a.t)
        elif a.kind == "bump":
            k = 0.22 * math.sin(math.pi * a.t)
            self.cam_pos = [a.p0[0] + a.dir[0] * k, a.p0[1] + a.dir[1] * k]
        if a.t >= 1.0:
            if a.kind == "turn":
                self.cam_h = -60.0 * self.heading   # snap, keeps the angle from drifting
            self.bob = 0.0
            self.anim = None

    def apply_camera(self):
        self.camera.setPos(self.cam_pos[0], self.cam_pos[1], EYE_HEIGHT + self.bob)
        self.camera.setHpr(self.cam_h, CAMERA_PITCH, 0)
        self.minimap.update(self.cam_pos, -self.cam_h)
        self.update_compass(-self.cam_h)

    # -- main loop ------------------------------------------------------------

    def update(self, task):
        dt = min(globalClock.getDt(), 0.1)  # noqa: F821  (ShowBase builtin)
        self.time += dt

        if self.fade is not None:
            self.step_fade(dt)
        elif self.anim is None:
            action = self.pending or self.held_action()
            self.pending = None
            if action:
                self.start_action(action)
        if self.anim is not None:
            self.step_anim(dt)
        self.apply_camera()
        self.animate_lights()

        if self.banner_time > 0:
            self.banner_time -= dt
            self.banner.setFg((0.95, 0.85, 0.65, max(0.0, min(1.0, self.banner_time))))
        return task.cont

    def animate_lights(self):
        t = self.time
        flicker = (1.0 + 0.07 * math.sin(t * 11.0) + 0.05 * math.sin(t * 23.0 + 1.3)
                   + 0.04 * math.sin(t * 37.0 + 0.7) + 0.03 * (random.random() - 0.5))
        r, g, b = self.torch_color
        self.torch.setColor((r * flicker * 1.6, g * flicker * 1.6, b * flicker * 1.6, 1))
        pulse = 0.8 + 0.2 * math.sin(t * 2.0)
        self.stairs_light.setColor((0.35 * pulse, 0.7 * pulse, 1.3 * pulse, 1))
        self.level.beacon.setH(t * 25.0)
        self.level.beacon.setZ(2.6 + 0.08 * math.sin(t * 1.5))

    def step_fade(self, dt):
        direction, t = self.fade
        t = min(1.0, t + dt / FADE_TIME)
        self.fade_card.setAlphaScale(t if direction == "out" else 1.0 - t)
        self.fade_card.setColor(0, 0, 0, 1)
        if t < 1.0:
            self.fade = [direction, t]
        elif direction == "out":
            self.load_level(self.depth + 1)
            self.fade = ["in", 0.0]
        else:
            self.fade = None
            self.fade_card.setColor(0, 0, 0, 0)

    # -- headless smoke test --------------------------------------------------

    def screenshot(self, prefix):
        self.taskMgr.remove("update")

        def shot(name):
            self.apply_camera()
            self.animate_lights()
            self.graphicsEngine.renderFrame()
            self.graphicsEngine.renderFrame()
            path = "%s_%s.png" % (prefix, name)
            self.win.saveScreenshot(Filename(path))
            print("wrote", path)

        def place(cell, heading):
            self.cell = cell
            self.heading = heading
            self.cam_pos = list(hg.hex_to_world(cell, CELL_SIZE))
            self.cam_h = -60.0 * heading
            self.reveal()

        dun = self.dungeon
        shot("start")
        place(self.cell, (self.heading + 1) % 6)
        shot("turned")
        if dun.halls:
            hall = max(dun.halls, key=len)
            centre = min(hall, key=lambda c: sum(hg.hex_distance(c, h) for h in hall))
            place(centre, 0)
            shot("hall")
        for d in range(6):
            n = hg.neighbor(dun.stairs, d)
            if dun.is_open(dun.stairs, d):
                place(n, hg.opposite(d))
                shot("stairs")
                break
        self.discovered = set(dun.cells)
        self.minimap.toggle()
        self.minimap.rebuild(self.discovered, self.visited)
        shot("map")
        sys.exit(0)


def main():
    game = Game()
    if SCREENSHOT_MODE:
        idx = sys.argv.index("--screenshot")
        prefix = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "shot"
        game.screenshot(prefix)
    game.run()


if __name__ == "__main__":
    main()
