import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Breakout Clone")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Game objects
class Paddle:
    def __init__(self):
        self.width = 100
        self.height = 15
        self.x = SCREEN_WIDTH // 2 - self.width // 2
        self.y = SCREEN_HEIGHT - 50
        self.speed = 8
        self.color = WHITE
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
    
    def move(self, direction):
        # Save original position
        old_x = self.x
        old_y = self.y
        
        if direction == "left" and self.x > 0:
            self.x -= self.speed
        if direction == "right" and self.x < SCREEN_WIDTH - self.width:
            self.x += self.speed
        if direction == "up" and self.y > 0:
            self.y -= self.speed
        if direction == "down" and self.y < SCREEN_HEIGHT - self.height:
            self.y += self.speed
            
        # Return the movement result
        return old_x, old_y

class Ball:
    def __init__(self):
        self.radius = 10
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed_x = random.choice([-4, -3, 3, 4])
        self.speed_y = -4
        self.color = WHITE
    
    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
    
    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y
        
        # Wall collisions
        if self.x <= self.radius or self.x >= SCREEN_WIDTH - self.radius:
            self.speed_x = -self.speed_x
        if self.y <= self.radius:
            self.speed_y = -self.speed_y
    
    def reset(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed_x = random.choice([-4, -3, 3, 4])
        self.speed_y = -4

class Brick:
    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.visible = True
    
    def draw(self):
        if self.visible:
            pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(screen, WHITE, (self.x, self.y, self.width, self.height), 1)

def create_bricks():
    bricks = []
    brick_rows = 6
    brick_cols = 10
    brick_width = 75
    brick_height = 20
    brick_padding = 5
    brick_offset_top = 60
    brick_offset_left = 30
    
    colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
    
    for row in range(brick_rows):
        for col in range(brick_cols):
            brick_x = col * (brick_width + brick_padding) + brick_offset_left
            brick_y = row * (brick_height + brick_padding) + brick_offset_top
            bricks.append(Brick(brick_x, brick_y, brick_width, brick_height, colors[row]))
    
    return bricks

def main():
    # Create game objects
    paddle = Paddle()
    ball = Ball()
    bricks = create_bricks()
    
    # Game variables
    clock = pygame.time.Clock()
    score = 0
    lives = 3
    game_over = False
    game_won = False

    # Font for text display
    font = pygame.font.Font(None, 36)
    
    # Main game loop
    while True:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        if not game_over and not game_won:
            # Paddle movement - using keyboard input
            keys = pygame.key.get_pressed()
            
            # Store original position to allow rollback if collision occurs
            old_paddle_x, old_paddle_y = paddle.x, paddle.y
            
            # Handle movement in separate steps for proper collision detection
            # First, check which arrow keys are pressed and move accordingly
            moved = False
            if keys[pygame.K_LEFT]:
                paddle.move("left")
                moved = True
            if keys[pygame.K_RIGHT]:
                paddle.move("right")
                moved = True
            if keys[pygame.K_UP]:
                paddle.move("up")
                moved = True
            if keys[pygame.K_DOWN]:
                paddle.move("down")
                moved = True
            
            # Check for collision with bricks after movement (only if there was movement)
            if moved:
                for brick in bricks:
                    if brick.visible:
                        # Check if the new paddle position overlaps with a brick
                        if (paddle.x < brick.x + brick.width and
                            paddle.x + paddle.width > brick.x and
                            paddle.y < brick.y + brick.height and
                            paddle.y + paddle.height > brick.y):
                            # Collision detected - revert to previous position
                            paddle.x, paddle.y = old_paddle_x, old_paddle_y
                            break  # Only one collision check per frame is needed for now

            # Move the ball
            ball.move()
            
            # Ball and paddle collision
            if (ball.y + ball.radius >= paddle.y and 
                ball.y - ball.radius <= paddle.y + paddle.height and
                ball.x >= paddle.x and 
                ball.x <= paddle.x + paddle.width):
                ball.speed_y = -abs(ball.speed_y)
                # Add some angle based on where the ball hits the paddle
                paddle_center = paddle.x + paddle.width // 2
                hit_pos = (ball.x - paddle_center) / (paddle.width // 2)
                ball.speed_x = hit_pos * 5
            
            # Ball and brick collisions
            for brick in bricks:
                if brick.visible:
                    if (ball.x + ball.radius >= brick.x and 
                        ball.x - ball.radius <= brick.x + brick.width and
                        ball.y + ball.radius >= brick.y and 
                        ball.y - ball.radius <= brick.y + brick.height):
                        brick.visible = False
                        ball.speed_y = -ball.speed_y
                        score += 10
                        break
            
            # Check if ball is out of bounds (loss of life)
            if ball.y > SCREEN_HEIGHT:
                lives -= 1
                if lives <= 0:
                    game_over = True
                else:
                    ball.reset()
            
            # Check if all bricks are destroyed (win condition)
            if all(not brick.visible for brick in bricks):
                game_won = True
        
        # Drawing
        screen.fill(BLACK)
        
        # Draw game objects
        paddle.draw()
        ball.draw()
        
        for brick in bricks:
            brick.draw()
        
        # Draw score and lives
        score_text = font.render(f"Score: {score}", True, WHITE)
        lives_text = font.render(f"Lives: {lives}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (SCREEN_WIDTH - 120, 10))
        
        # Draw game over or win messages
        if game_over:
            game_over_text = font.render("Game Over! Press R to restart", True, WHITE)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(game_over_text, text_rect)
        
        if game_won:
            win_text = font.render("You Win! Press R to restart", True, WHITE)
            text_rect = win_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(win_text, text_rect)
        
        # Restart game with 'R' key
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r] and (game_over or game_won):
            # Reset game state
            paddle = Paddle()
            ball = Ball()
            bricks = create_bricks()
            score = 0
            lives = 3
            game_over = False
            game_won = False
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()