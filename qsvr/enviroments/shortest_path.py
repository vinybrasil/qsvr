import numpy as np


class ShortestPath:
    def __init__(self, cost_matrix, source=0, target=8):
        self.cost_matrix = cost_matrix
        self.source = source
        self.target = target
        self.state = source
        self.done = False
        self.observation_space = cost_matrix.shape[0]
        self.action_space = cost_matrix.shape[0]
        self.visited = set()

    def calculate_cost(self, path):
        # total_cost = 0
        costs = []
        for i in range(len(path) - 1):
            # total_cost += self.cost_matrix[path[i]][path[i+1]]
            costs.append(self.cost_matrix[path[i]][path[i + 1]])
        return costs

    def calculate_reward(self, costs):
        reward = 0
        for i in range(len(costs)):
            # reward += 1/costs[i]
            reward -= costs[i]

        return reward

    def get_valid_actions(self, new_state=None):
        # non_zeros = np.nonzero(self.cost_matrix[self.state])[0]

        # return list(set(list(non_zeros)) - self.visited)
        if new_state is not None:
            valid_actions = np.where(np.array(self.cost_matrix[new_state]) > 0)[0]
            return valid_actions   
        valid_actions = np.where(np.array(self.cost_matrix[self.state]) > 0)[0]
        return valid_actions

    def reset(self):
        self.done = False
        self.state = self.source
        self.visited = set()
        return self.state

    def check_dead(self):
        # if len(self.get_valid_actions()) == 0:
        #    self.done = True
        if self.state == self.target:
            self.done = True

    def step(self, action):
        valid_actions = self.get_valid_actions()
        # if action not in valid_actions:
        #    raise

        # self.visited.add(self.state)
        reward = self.calculate_reward(self.calculate_cost([self.state, action]))
        self.state = action

        # if action == self.target:
        #    self.done = True

        self.check_dead()

        # if self.state == self.target:
        #    reward += 100

        return self.state, self.done, reward
