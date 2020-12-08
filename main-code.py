"""
This project aims to recreate the classic snake game with Pygame and then teach an AI to play it using NEAT.

I might also explore other RL approaches if NEAT goes well
"""

import warnings
import pygame
import sys
import random
import neat
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

# Window size
frame_size_x = 200
frame_size_y = 200

clock = pygame.time.Clock()

# Checks for errors encountered
check_errors = pygame.init()
# pygame.init() example output -> (6, 0)
# second number in tuple gives number of errors
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')

# Initialise game window
pygame.display.set_caption('Snake Eater')
game_window = pygame.display.set_mode((frame_size_x, frame_size_y))

# Colors (R, G, B)
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

# FPS (frames per second) controller
fps_controller = pygame.time.Clock()
render_graphics = False


def show_score(choice, color, font, size, age_tracker):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(age_tracker), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (frame_size_x / 10 + 15, 5)
    else:
        score_rect.midtop = (frame_size_x / 2, frame_size_y / 1.25)
    game_window.blit(score_surface, score_rect)


class Snake:
    def __init__(self):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100 - 10, 50], [100 - (2 * 10), 50]]
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.age = 0


class Food:
    def __init__(self):
        self.food_pos = [random.randrange(1, (frame_size_x // 10)) * 10, random.randrange(1, (frame_size_y // 10)) * 10]
        self.food_spawn = True


def train_snake(genomes, config):
    nets = []
    ge = []
    snakes = []
    foods = []
    age_tracker = 0

    for _, g in genomes:  # Create all genomes, neural nets, and snakes
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        snakes.append(Snake())
        foods.append(Food())
        g.fitness = 0
        ge.append(g)

    # Main logic
    run = True
    while run and len(snakes) > 0:
        if render_graphics:
            clock.tick(50)
        age_tracker += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        for x, snake in enumerate(snakes):
            # ge[x].fitness += 0.05
            # Reset inputs
            danger_left = 0
            danger_right = 0
            danger_ahead = 0
            direction_up = 0
            direction_down = 0
            direction_left = 0
            direction_right = 0
            food_up = 0
            food_down = 0
            food_left = 0
            food_right = 0

            # Left wall dangers
            if snake.snake_pos[0] < 10 and snake.direction == 'LEFT':
                danger_ahead = 1
            if snake.snake_pos[0] < 10 and snake.direction == 'UP':
                danger_left = 1
            if snake.snake_pos[0] < 10 and snake.direction == 'DOWN':
                danger_right = 1

            # Right wall dangers
            if snake.snake_pos[0] > frame_size_x - 20 and snake.direction == 'RIGHT':
                danger_ahead = 1
            if snake.snake_pos[0] > frame_size_x - 20 and snake.direction == 'UP':
                danger_right = 1
            if snake.snake_pos[0] > frame_size_x - 20 and snake.direction == 'DOWN':
                danger_left = 1

            # Top wall dangers
            if snake.snake_pos[1] < 10 and snake.direction == 'LEFT':
                danger_right = 1
            if snake.snake_pos[1] < 10 and snake.direction == 'RIGHT':
                danger_left = 1
            if snake.snake_pos[1] < 10 and snake.direction == 'UP':
                danger_ahead = 1

            # Bottom wall dangers
            if snake.snake_pos[1] > frame_size_y - 20 and snake.direction == 'LEFT':
                danger_left = 1
            if snake.snake_pos[1] > frame_size_y - 20 and snake.direction == 'RIGHT':
                danger_right = 1
            if snake.snake_pos[1] > frame_size_y - 20 and snake.direction == 'DOWN':
                danger_ahead = 1

            if snake.direction == 'UP':
                direction_up = 1
            if snake.direction == 'DOWN':
                direction_down = 0
            if snake.direction == 'LEFT':
                direction_left = 1
            if snake.direction == 'RIGHT':
                direction_right = 0

            if snake.snake_pos[1] < foods[x].food_pos[1]:
                food_up = 1
            if snake.snake_pos[1] > foods[x].food_pos[1]:
                food_down = 0
            if snake.snake_pos[0] < foods[x].food_pos[0]:
                food_left = 1
            if snake.snake_pos[0] < foods[x].food_pos[0]:
                food_right = 0

            max_dist = math.sqrt(frame_size_x ** 2 + frame_size_y)
            min_dist = -1 * max_dist
            food_dist = (snake.snake_pos[0] - foods[x].food_pos[0]) + (snake.snake_pos[1] - foods[x].food_pos[1])
            food_dist_norm = (food_dist - min_dist) / (max_dist - min_dist)

            nn_inputs = [danger_left, danger_right, danger_ahead, direction_up, direction_down, direction_left,
                         direction_right, food_up, food_down, food_left, food_right, food_dist_norm]

            # nn_inputs = [snake.snake_pos[0], snake.snake_pos[1], foods[x].food_pos[0], foods[x].food_pos[1]]

            # print(nn_inputs)

            output = nets[x].activate(nn_inputs)

            max_value = max(output)
            max_index = output.index(max_value)

            if max_index == 0:
                snake.change_to = 'UP'
            elif max_index == 1:
                snake.change_to = 'DOWN'
            elif max_index == 2:
                snake.change_to = 'LEFT'
            elif max_index == 3:
                snake.change_to = 'RIGHT'

            # Making sure the snake cannot move in the opposite direction instantaneously
            if snake.change_to == 'UP' and snake.direction != 'DOWN':
                snake.direction = 'UP'
            if snake.change_to == 'DOWN' and snake.direction != 'UP':
                snake.direction = 'DOWN'
            if snake.change_to == 'LEFT' and snake.direction != 'RIGHT':
                snake.direction = 'LEFT'
            if snake.change_to == 'RIGHT' and snake.direction != 'LEFT':
                snake.direction = 'RIGHT'

            # Moving the snake
            if snake.direction == 'UP':
                snake.snake_pos[1] -= 10
            if snake.direction == 'DOWN':
                snake.snake_pos[1] += 10
            if snake.direction == 'LEFT':
                snake.snake_pos[0] -= 10
            if snake.direction == 'RIGHT':
                snake.snake_pos[0] += 10

            # Give some fitness if moving in the right direction
            # if snake.direction == 'UP' and food_up:
            #     ge[x].fitness += 0.001
            # if snake.direction == 'DOWN' and food_down:
            #     ge[x].fitness += 0.001
            # if snake.direction == 'LEFT' and food_left:
            #     ge[x].fitness += 0.001
            # if snake.direction == 'RIGHT' and food_right:
            #     ge[x].fitness += 0.001

            # Snake body grow and increase age
            snake.age += 1
            snake.snake_body.insert(0, list(snake.snake_pos))
            if snake.snake_pos[0] == foods[x].food_pos[0] and snake.snake_pos[1] == foods[x].food_pos[1]:
                ge[x].fitness += 5
                foods[x].food_spawn = False
            else:
                snake.snake_body.pop()

            # Spawning food on the screen
            if not foods[x].food_spawn:
                foods[x].food_pos = [random.randrange(1, (frame_size_x // 10)) * 10,
                                     random.randrange(1, (frame_size_y // 10)) * 10]
            foods[x].food_spawn = True

            # Game Over conditions
            # Getting out of bounds
            if snake.snake_pos[0] < 0 or snake.snake_pos[0] > frame_size_x - 10:
                nets.pop(x)
                ge.pop(x)
                snakes.pop(x)
                foods.pop(x)
            elif snake.snake_pos[1] < 0 or snake.snake_pos[1] > frame_size_y - 10:
                nets.pop(x)
                ge.pop(x)
                snakes.pop(x)
                foods.pop(x)
            # Living for too long
            elif snake.age > 500:
                nets.pop(x)
                ge.pop(x)
                snakes.pop(x)
                foods.pop(x)
            # Touching the snake body
            for block in snake.snake_body[1:]:
                if snake.snake_pos[0] == block[0] and snake.snake_pos[1] == block[1]:
                    try:
                        nets.pop(x)
                        ge.pop(x)
                        snakes.pop(x)
                        foods.pop(x)
                    except:
                        pass

        # GFX
        if render_graphics:
            game_window.fill(black)
            for snake in snakes:
                for pos in snake.snake_body:
                    pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))

            # Snake food
            for food in foods:
                pygame.draw.rect(game_window, white, pygame.Rect(food.food_pos[0], food.food_pos[1], 10, 10))

            # Refresh game screen
            show_score(1, white, 'calibri', 20, age_tracker)
            pygame.display.update()


def run_snake(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(train_snake, 1000)
    print('\nBest genome:\n{!s}'.format(winner))
    plot_stats(stats)


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-snake.txt")
    run_snake(config_path)
