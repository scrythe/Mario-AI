import retro
import neat
import os
import cv2
import numpy


def start_game(genome: neat.DefaultGenome, net: neat.nn.RecurrentNetwork):
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1.state')
    screen = env.reset()

    width, height, color = env.observation_space.shape
    scaled_width = int(width/8)
    scaled_height = int(height/8)
    # cv2.namedWindow("main", cv2.WINDOW_NORMAL)

    done = False
    while not done:
        env.render()

        resized_screen = cv2.resize(screen, (scaled_width, scaled_height))
        recolored_screen = cv2.cvtColor(resized_screen, cv2.COLOR_BGR2GRAY)
        reshaped_screen = numpy.reshape(
            recolored_screen, (scaled_width, scaled_height))
        # cv2.imshow('main', recolored_screen)
        # cv2.waitKey(1)

        action = env.action_space.sample()
        screen, reward, done, info = env.step(action)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)
        start_game(genome, net)


def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    winner = population.run(eval_genomes)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join('config-feedforward')
    run_neat(config_path)
