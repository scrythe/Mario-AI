import retro
import neat
import os


def start_game():
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1.state')
    obs = env.reset()
    done = False
    while not done:
        env.render()


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        start_game()


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
