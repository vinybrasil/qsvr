from dataclasses import dataclass

import numpy as np


@dataclass
class ChainWalk:
    # initial_state = 0
    state = 0
    num_steps = 0
    done = False
    action_space = 2
    observation_space = 4
    threshold = 0.90

    def set_seed(self, seed):
        np.random.seed(seed)

    def check_end(self):
        if self.num_steps == 4:
            self.num_steps = 0
            self.done = True

    def swap_action(self, action):
        rn = np.random.rand()
        if rn > self.threshold:
            action = np.abs(action - 1)
        return action

    def step(self, action):

        action = self.swap_action(
            action
        )  # tem uma probabilidade de 0.1 de mudar de acao

        if self.state == 0 and action == 0:
            self.state = 0
            self.num_steps += 1
            # return self.state, 0
            self.check_end()
            # return self.state, 0, True
            return self.state, 0, self.done

        if self.state == 0 and action == 1:
            self.state = 1
            self.num_steps += 1
            # return self.state, 1
            self.check_end()
            # return self.state, 1, True
            return self.state, 0, self.done

        if self.state == 1 and action == 0:
            self.state = 0
            self.num_steps += 1
            # return self.state, 1
            self.check_end()

            # return self.state, 1, True
            return self.state, 0, self.done

        if self.state == 1 and action == 1:
            self.state = 2
            self.num_steps += 1
            # return self.state, 1
            self.check_end()
            # return self.state, 1, True
            return self.state, 1, self.done

        if self.state == 2 and action == 0:
            self.state = 1
            self.num_steps += 1
            # return self.state, 1
            self.check_end()
            # return self.state, 1, True
            return self.state, 1, self.done

        if self.state == 2 and action == 1:
            self.state = 3
            self.num_steps += 1
            # return self.state, 0
            self.check_end()
            # return self.state, 0, True
            return self.state, 0, self.done


        if self.state == 3 and action == 0:
            self.state = 2
            self.num_steps += 1
            # return self.state, 0
            self.check_end()
            # return self.state, 0, True
            return self.state, 0, self.done

        if self.state == 3 and action == 1:
            self.state = 3
            self.num_steps += 1
            self.check_end()
            # return self.state, 0, True
            return self.state, 0, self.done

    def reset(self):
        self.state = 0
        self.done = False
        return self.state
