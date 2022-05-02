import neat
import retro
import cv2
import numpy


class Game:
    def __init__(self, genome: neat.DefaultGenome, config):
        self.genome = genome
        self.net = neat.nn.RecurrentNetwork.create(genome, config)
        self.env = retro.make(game='SuperMarioWorld-Snes',
                              state='YoshiIsland1.state', scenario='scenario.json', info='data.json')
        self.screen = self.env.reset()
        self.done = False
        self.data = self.env.data
        self.data.__setitem__('lives', 1)

        self.last_time_reward = 0
        self.current_fitness = 0
        self.previous_fitness = 0
        self.xpos = 0
        self.previous_xpos = 0
        # cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        width, height, color = self.env.observation_space.shape
        self.scaled_width = int(width/8)
        self.scaled_height = int(height/8)

    def get_actions(self):
        reshaped_screen = self.rescale_screen()
        img_array = numpy.ndarray.flatten(reshaped_screen)
        actions = self.net.activate(img_array)
        return actions

    def run(self):
        # self.env.render()
        actions = self.get_actions()
        self.screen, reward, self.done, info = self.env.step(actions)
        self.xpos = info['x']
        endOfLevel = info['x']

        if endOfLevel == 1:
            self.current_fitness += 100000

        self.current_fitness += reward
        if self.xpos > self.previous_xpos:
            self.current_fitness += 1
            self.previous_xpos = self.xpos

        if self.current_fitness > self.previous_fitness:
            self.last_time_reward = 0
            self.previous_fitness = self.current_fitness
        else:
            self.last_time_reward += 1

        if self.last_time_reward > 250:
            self.done = True

    def rescale_screen(self):
        resized_screen = cv2.resize(
            self.screen, (self.scaled_width, self.scaled_height))
        recolored_screen = cv2.cvtColor(resized_screen, cv2.COLOR_BGR2GRAY)
        reshaped_screen = numpy.reshape(
            recolored_screen, (self.scaled_width, self.scaled_height))

        # cv2.imshow('main', recolored_screen)
        # cv2.waitKey(1)

        return reshaped_screen
