# Space Sim

A small planet-to-space flight sim in Panda3D. You start parked on a landing pad on a
spherical planet with mountains, valleys, seas and lakes. Take off, fly around, climb out
of the atmosphere into space, then turn around and land anywhere: on a hilltop, a beach or
the water.

Everything is generated at startup: the terrain (numpy fractal noise on a cube-sphere with
quadtree level of detail), the ocean, the scattering atmosphere and sky, the ship and the pad.

## Run

```
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

Headless smoke test (renders a few views offscreen to `shot_*.png`):

```
.venv/bin/python main.py --screenshot shot
```

## Controls

| Input | Action |
|---|---|
| Mouse / Arrow keys | Pitch & yaw |
| Q / E | Roll |
| W / S | Throttle up / down |
| X | Cut throttle |
| Space / C (or Ctrl) | Thrust up / down |
| A / D | Strafe |
| Shift | Boost |
| F | Toggle flight assist |
| V | Camera: chase / far chase / cockpit |
| R | Back to the pad |
| H | Hide help |
| Esc | Quit |

## Flying

- **Flight assist ON** (default): throttle sets your target forward speed, and the thrusters
  cancel drift and hold you against gravity. Let go of everything and the ship hovers. Hold
  **C** to descend: the descent slows automatically as you get close to the ground.
- **Flight assist OFF**: Newtonian flight. Throttle is raw engine thrust and gravity is
  yours to handle. At about 440 m/s sideways near the surface you are in orbit (the HUD
  shows the local orbital speed).
- Space starts at 12 km. Air drag fades with altitude, so you go much faster up there.
  Hold **Shift** to boost out of the atmosphere.
- **Landing**: touch down on your landing legs, roughly level (within 30°), slower than
  6 m/s vertically and 8 m/s horizontally. Anything harder is a crash, and you respawn on
  the pad.
- The blue `[ PAD ]` marker shows the way home.

## Files

- `main.py`: game, rendering, shaders, LOD, ship physics
- `terrain.py`: the planet's height function (shared by the renderer and the collision code)
