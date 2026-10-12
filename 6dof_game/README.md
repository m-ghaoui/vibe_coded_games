# Quake6

A small 6DoF shooter in Panda3D. The level, textures and models are all generated when the game starts.

## Run

```
.venv/bin/python main.py
```

Headless smoke test (renders offscreen and saves a screenshot):

```
.venv/bin/python main.py --screenshot shot.png
```

## Controls

| Input | Action |
|---|---|
| Mouse | Look (pitch/yaw relative to your current orientation) |
| W A S D | Move forward / strafe |
| Space / Ctrl | Move up / down |
| Q / E | Roll |
| Shift | Boost |
| Left mouse | Fire plasma bolts |
| F | Toggle headlight |
| G | Toggle shadows |
| R | Restart (respawns you and all drones) |
| Esc | Quit |

## Drones

Drones hover around their post until they see you (line of sight, up to 70 units). Then they:

- keep about 16 units away and strafe from side to side
- aim where you'll be when the shot arrives
- glow before each shot, which is your cue to dodge

If they lose sight of you, they fly to where they last saw you and search for a while. Shooting a drone, or getting hit, also brings nearby drones over. Drones take 3 hits. You have 100 health, which slowly regenerates after 5 seconds without taking damage.
