# Hex Dungeon Crawl

A step-and-turn first-person dungeon crawler in Panda3D, played on a **hexagonal grid**
instead of a square one. Every chamber is a hexagon. You turn in 60° steps, and you can
step to any of the six neighbouring cells without turning: forward, back, and the four
diagonals.

Each level is a maze carved inside a large hex-shaped region, with loops and a few
pillared halls. Find the glowing stairs to go deeper. Every depth is bigger and has its
own stone tint and fog colour. You carry a flickering torch, so you only see a few cells
ahead. The minimap fills in as you explore.

Everything is generated at startup: the maze, the geometry, and the stone textures and
normal maps (built from noise).

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

| Key | Action |
|---|---|
| W / ↑ | Step forward |
| S / ↓ | Step back |
| A / ← , D / → | Turn 60° left / right |
| Q / E | Step forward-left / forward-right |
| Z / C | Step back-left / back-right |
| Space | Descend (on the stairs) |
| M | Toggle the big map |
| N | Regenerate the current level |
| H | Hide help |
| Esc | Quit |

Hold a key to keep walking or turning.

## The hex compass

The six headings are **N, NE, SE, S, SW, NW**. The hex compass at the bottom of the
screen turns with you, and the pointer on top shows which way you face. Q/E/Z/C step
toward the compass labels on either side of your heading without changing where you look.

## Files

- `main.py`: game loop, input, camera animation, lighting, HUD, compass, level transitions
- `hexgrid.py`: hex coordinates, maze/hall generation, visibility (pure Python)
- `dungeon_mesh.py`: procedural textures and level geometry
- `minimap.py`: fog-of-war hex map in its own display region
