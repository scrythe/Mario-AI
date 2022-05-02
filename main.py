import neat
import os
from game import Game
import pickle


def eval_genome(genome, config):
    # genome: neat.DefaultGenome = genomes[0][1]
    genome: neat.DefaultGenome
    game = Game(genome, config)
    while not game.done:
        game.run()
    return game.current_fitness


def run_neat(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # population = neat.Population(config)
    population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-69')

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))

    parallel = neat.ParallelEvaluator(
        6, eval_genome)
    winner = population.run(parallel.evaluate)
    # winner = population.run(eval_genome)
    with open('best.genome', 'wb') as file:
        pickle.dump(winner, file)


def run_best_genome(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    with open('best.genome', 'rb') as file:
        winner = pickle.load(file)
    eval_genome(winner, config)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join('config-feedforward')
    run_neat(config_path)
    # run_best_genome(config_path)
