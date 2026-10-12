# Isometric RPG

An isometric field where a warrior fights skeletons, made with Panda3D. All geometry is built in code when the program starts, so there are no asset files.

## Run

```
.venv/bin/python main.py
```

Headless smoke test (renders offscreen and saves a screenshot):

```
.venv/bin/python main.py --screenshot shot.png [--frames 150] [--zoom 5] [--keys wd] [--attack] [--spawn-near]
```

| Option | Effect |
|---|---|
| `--frames N` | Frame to capture (default 30, at a fixed 60 fps) |
| `--zoom` | How many world units tall the view is (default 7) |
| `--time` | Shifts the animation clock |
| `--keys` | Movement keys held down, e.g. `wd` |
| `--attack` | Swing the sword continuously |
| `--spawn-near` | Start with three skeletons close by |

## Controls

| Input | Action |
|---|---|
| W A S D / arrow keys | Run (W runs up the screen) |
| Left click | Swing the sword toward the mouse cursor (hold to keep swinging) |
| Mouse wheel | Zoom |
| R | Rise again after dying |
| Esc | Quit |

## Skeletons

Skeletons climb out of the ground and wander until you come within 7 units. Then they walk over, raise their sword (your cue to back off or strike first) and chop. Each hit costs 12 health. They take 3 hits to kill and fall apart into a pile of bones. Hitting a skeleton interrupts its attack.

There are 4 skeletons at a time at first. Every 3 kills adds one more, up to 8. After 4 seconds without taking damage, your health slowly regenerates.

## Code

| File | Contents |
|---|---|
| `main.py` | Game loop, input, camera, combat, spawning, HUD |
| `warrior.py` | Warrior model, cloak, idle/run/swing/death animation |
| `skeleton.py` | Skeleton model, animation and AI |
| `field.py` | Ground, grass, rocks, trees and collision |
| `geometry.py` | Procedural mesh helpers |
