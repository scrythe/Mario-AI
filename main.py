import neat
import os
from game import Game
import retro
import pickle


def eval_genome(genome, config):
    genome: neat.DefaultGenome
    genome.fitness = 0
    game = Game(genome, config)
    while not game.done:
        game.run()
    return genome.fitness


def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    parallel = neat.ParallelEvaluator(
        6, eval_genome)
    winner = population.run(parallel.evaluate)
    with open('best.genome', 'wb') as file:
        pickle.dump(winner, file)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join('config-feedforward')
    run_neat(config_path)
