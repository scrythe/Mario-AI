import neat
import retro
import cv2
import numpy


class Game:
    def __init__(self, genome: neat.DefaultGenome, config):
        self.genome = genome
        self.net = neat.nn.RecurrentNetwork.create(genome, config)
        self.env = retro.make(game='SuperMarioWorld-Snes',
                              state='YoshiIsland1.state')
        self.screen = self.env.reset()
        self.done = False
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        width, height, color = self.env.observation_space.shape
        self.scaled_width = int(width/8)
        self.scaled_height = int(height/8)

    def run(self):
        self.env.render()
        reshaped_screen = self.rescale_screen()
        action = self.env.action_space.sample()
        self.screen, reward, self.done, info = self.env.step(action)

    def rescale_screen(self):
        resized_screen = cv2.resize(
            self.screen, (self.scaled_width, self.scaled_height))
        recolored_screen = cv2.cvtColor(resized_screen, cv2.COLOR_BGR2GRAY)
        reshaped_screen = numpy.reshape(
            recolored_screen, (self.scaled_width, self.scaled_height))

        # cv2.imshow('main', recolored_screen)
        # cv2.waitKey(1)

        return reshaped_screen
