import numpy as np


class RandomWalk:
    def __init__(self, observation_space, max_steps=None):
        self.action_space = 2
        self.observation_space = observation_space
        self.max_steps = max_steps
        self.done = False
        if max_steps is None:
            self.max_steps = observation_space
        self.state = int(np.ceil(self.observation_space / 2))
        self.num_steps = 0

    def reset(self):
        self.state = int(np.ceil(self.observation_space / 2))
        self.done = False
        self.num_steps = 0
        return self.state

    def check_end(self):
        if (
            (self.num_steps == self.max_steps)
            or (self.state == self.observation_space)
            or (self.state == 1)
        ):
            self.num_steps = 0
            self.done = True

    def reward(self):
        if self.state == self.observation_space:
            return 10
        if self.state == 1:
            return -10
        return 0

    def step(self, action):
        # 1 vai pra direita, -1 vai pra esquerda
        if action not in [0, 1]:
            raise
        if action == 0:
            action = -1

        self.state += action
        self.num_steps += 1
        self.check_end()
        reward = self.reward()

        # print(self.num_steps)
        return self.state, reward, self.done
