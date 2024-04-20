import random
import time
import pygame
import sys

# Define some colors
black     = ( 0, 0, 0)
white     = (255, 255, 255)
gray     = (150, 150, 150)
red       = (255,   0,   0)

# Set the width and height of the screen (width, height).
size = (700, 500)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("Snake Game")

class Snake:
    def __init__(self):
        self.body = [(100, 50), (90, 50), (80, 50)]
        self.direction = "Right"

    def move(self):
        if self.direction == "Right":
            self.body.insert(0, (self.body[0][0] + 10, self.body[0][1]))
            self.body.pop()
        elif self.direction == "Left":
            self.body.insert(0, (self.body[0][0] - 10, self.body[0][1]))
            self.body.pop()
        elif self.direction == "Up":
            self.body.insert(0, (self.body[0][0], self.body[0][1] - 10))
            self.body.pop()
        elif self.direction == "Down":
            self.body.insert(0, (self.body[0][0], self.body[0][1] + 10))
            self.body.pop()

    def eat(self):
        if self.body[0] in food.position:
            self.body.insert(0, self.body[0])
            return True
        else:
            return False

class Food:
    def __init__(self):
        self.position = [(random.randint(0, 69) * 10, random.randint(0, 49) * 10)]

    def generate(self):
        self.position = [(random.randint(0, 69) * 10, random.randint(0, 49) * 10)]

snake = Snake()
food = Food()

# -------- Main Program Loop -----------
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT and snake.direction != "Right":
                snake.direction = "Left"
            elif event.key == pygame.K_RIGHT and snake.direction != "Left":
                snake.direction = "Right"
            elif event.key == pygame.K_UP and snake.direction != "Down":
                snake.direction = "Up"
            elif event.key == pygame.K_DOWN and snake.direction != "Up":
                snake.direction = "Down"

    snake.move()
    if not snake.eat():
        for segment in snake.body[1:]:
            if segment in snake.body[:1]:
                print("Game Over!")
                pygame.quit()
                sys.exit()

    screen.fill(black)

    for position in snake.body:
        pygame.draw.rect(screen, gray, [position[0], position[1], 10, 10])

    pygame.draw.rect(screen, white, food.position[0] + (10, 10), 10)

    pygame.display.flip()
    clock = pygame.time.Clock()
    clock.tick(60)