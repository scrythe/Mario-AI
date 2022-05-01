import retro


def start_game():
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland1.state')
    env.reset()
    done = False
    while not done:
        env.render()


if __name__ == '__main__':
    start_game()
