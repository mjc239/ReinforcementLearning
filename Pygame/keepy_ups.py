import pygame
import numpy as np

# Variables
WIDTH = 500
HEIGHT = 500
BORDER = 20
VELOCITY = 1
FRAME_RATE = 100
BOUNCE_VEL_DIVISOR = 50
RANDOM_VEL_SIZE = 2
INPUT_MODE = 'keyboard'
assert INPUT_MODE in ['keyboard', 'mouse']
KEYBOARD_SPEED = 5

bg_color = pygame.Color('black')
wall_color = pygame.Color('white')


# Define objects
class Ball:
    RADIUS = 20
    BALL_COLOR = pygame.Color('blue')
    MAX_X = WIDTH - BORDER - RADIUS
    MIN_X = BORDER + RADIUS
    MIN_Y = BORDER + RADIUS

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.bounce_count = 0

    def show(self, color):
        global screen
        pygame.draw.circle(screen, color, (self.x, self.y), self.RADIUS)

    def update(self, paddle):
        global bg_color

        # Update position
        self.show(bg_color)
        self.x += self.vx
        self.y += self.vy
        # Push back in bounds if in wall
        self.x = max(min(self.x, self.MAX_X), self.MIN_X)
        self.y = max(self.y, self.MIN_Y)
        self.show(self.BALL_COLOR)

        # Update velocity
        global BORDER, WIDTH, HEIGHT
        hit_left_wall = self.x <= self.RADIUS + BORDER
        hit_right_wall = self.x + self.RADIUS + BORDER >= WIDTH
        hit_top_wall = self.y <= self.RADIUS + BORDER
        hit_paddle = self.y >= HEIGHT - paddle.PADDLE_HEIGHT - self.RADIUS \
                     and ((paddle.x - paddle.PADDLE_WIDTH//2) <= self.x <= (paddle.x + paddle.PADDLE_WIDTH//2))
        if hit_left_wall or hit_right_wall:
            self.vx *= -1
        if hit_top_wall or hit_paddle:
            self.vy *= -1

        # Add random velocity when paddle is hit
        global BOUNCE_VEL_DIVISOR, RANDOM_VEL_SIZE
        if hit_paddle:
            self.vx += round((self.x - paddle.x)/BOUNCE_VEL_DIVISOR) + \
                       int(np.random.choice(np.arange(-RANDOM_VEL_SIZE, RANDOM_VEL_SIZE+1)))
            self.vy -= 1
            self.bounce_count += 1

    def game_over(self):
        global HEIGHT
        return self.y + self.RADIUS > HEIGHT


class Paddle:
    PADDLE_HEIGHT = 20
    PADDLE_WIDTH = 100
    PADDLE_COLOR = pygame.Color('red')
    MIN_X = BORDER + PADDLE_WIDTH//2
    MAX_X = WIDTH - BORDER - PADDLE_WIDTH//2

    def __init__(self, x):
        self.x = x

    def show(self, color):
        global WIDTH, HEIGHT
        pygame.draw.rect(screen,
                         color,
                         pygame.Rect((self.x - self.PADDLE_WIDTH//2, HEIGHT-self.PADDLE_HEIGHT),
                                     (self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
                         )

    def update(self):
        global bg_color, INPUT_MODE, KEYBOARD_SPEED
        self.show(bg_color)
        if INPUT_MODE == 'keyboard':
            keys = pygame.key.get_pressed()
            self.x += (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])*KEYBOARD_SPEED
        elif INPUT_MODE == 'mouse':
            self.x = pygame.mouse.get_pos()[0]
        self.x = max(min(self.x, self.MAX_X), self.MIN_X)
        self.show(self.PADDLE_COLOR)


# Create objects
ball = Ball(WIDTH//2, 9*HEIGHT//10, 0, -VELOCITY)
game_paddle = Paddle(WIDTH//2)

# Initialise game
pygame.init()

# Draw game screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(bg_color)
pygame.draw.rect(screen, wall_color, pygame.Rect((0, 0), (WIDTH, BORDER)))
pygame.draw.rect(screen, wall_color, pygame.Rect((0, 0), (BORDER, HEIGHT)))
pygame.draw.rect(screen, wall_color, pygame.Rect((WIDTH-BORDER, 0), (BORDER, HEIGHT)))

# Text objects
font = pygame.font.SysFont('Arial', size=32)

clock = pygame.time.Clock()

while True:
    pygame.display.flip()
    e = pygame.event.poll()
    if e.type == pygame.QUIT:
        break
    clock.tick(FRAME_RATE)

    current_bounce = ball.bounce_count
    if ball.game_over():
        gameover = font.render(f'Game Over! Score: {current_bounce}', False, (255, 0, 0))
        screen.blit(gameover, (WIDTH//5, HEIGHT//2))
        continue
    else:
        game_paddle.update()
        ball.update(game_paddle)
        if ball.bounce_count > current_bounce:
            pygame.draw.rect(screen, bg_color, pygame.Rect((BORDER, BORDER), (WIDTH-2*BORDER, HEIGHT//2)))
            game_counter = font.render(f'{ball.bounce_count}', False, (255, 0, 0))
            screen.blit(game_counter, (2*BORDER, 2*BORDER))

pygame.quit()
