import neat
import os
from game import Game


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        game = Game(genome, config)
        while not game.done:
            game.run()


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
