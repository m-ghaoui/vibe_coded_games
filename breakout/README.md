# Breakout Clone

A classic Breakout (Arkanoid) game implementation using Python and Pygame.

This game was created with the assistance of Qwen, an AI coding assistant. The implementation was generated through a collaboration between the user's requirements and Qwen's programming capabilities.

## Features

- Player-controlled paddle at the bottom of the screen
- Ball physics with wall, paddle, and brick collisions
- Colorful bricks arranged in rows
- Score tracking
- Lives system (3 lives)
- Win/lose conditions
- Restart functionality

## Controls

- Left Arrow Key: Move paddle left
- Right Arrow Key: Move paddle right
- R Key: Restart game after game over or win

## Requirements

- Python 3.x
- Pygame library

## How to Run

1. Ensure you have a Python virtual environment with Pygame installed
2. Navigate to this directory
3. Run: `python main.py`

## Game Rules

- Break all the bricks to win the game
- Don't let the ball fall below the paddle (lose a life)
- You have 3 lives to complete the game
- Press 'R' at any time to restart the game after winning or losing

## Implementation Details

The game is built using object-oriented Python with three main classes:
- Paddle: Player-controlled object at the bottom
- Ball: Physics-based object that bounces around the screen
- Brick: Colorful objects that disappear when hit by the ball

## About This Implementation

This Breakout clone was created using Qwen, an AI coding assistant. The implementation process involved:
1. Understanding the requirements for a classic Breakout game
2. Designing the game architecture with appropriate classes and methods
3. Implementing the core game mechanics including physics, collision detection, and scoring
4. Adding visual elements and user interface components
5. Creating a complete, playable game that can be run from the command line

The AI assistant helped design and implement all aspects of this classic arcade game while maintaining code quality and adherence to Python best practices.

## System Information

This project was developed using:
- Qwen as the primary coding AI assistant
- Ollama for local language model access during development
- RTX 4070 GPU for accelerated computing tasks (when applicable)