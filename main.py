import neat
import os
from game import Game
import retro


def eval_genomes(genomes, config, env):
    for genome_id, genome in genomes:
        genome: neat.DefaultGenome
        game = Game(genome, config, env)
        while not game.done:
            game.run()
        genome.fitness = 0


def run_neat(config_file, env):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))
    winner = population.run(
        lambda genomes, config: eval_genomes(genomes, config, env))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join('config-feedforward')
    env = retro.make(game='SuperMarioWorld-Snes',
                          state='YoshiIsland1.state', scenario='scenario.json')
    run_neat(config_path, env)
