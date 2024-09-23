import json
import os
import pickle
import random

import numpy as np


class Sampler:

    def load_db(self, fld):

        self.db = pickle.load(open(os.path.join(fld, "db.pickle"), "rb"))
        param = json.load(open(os.path.join(fld, "param.json"), "rb"))
        self.i_db = 0
        self.n_db = param["n_episodes"]
        self.sample = self.__sample_db
        for attr in param:
            if hasattr(self, attr):
                setattr(self, attr, param[attr])
        self.title = "DB_" + param["title"]

    def build_db(self, n_episodes, fld):
        db = []
        for i in range(n_episodes):
            prices, title = self.sample()
            db.append((prices, "[%i]_" % i + title))
        os.makedirs(fld)  # don't overwrite existing fld
        pickle.dump(db, open(os.path.join(fld, "db.pickle"), "wb"))
        param = {"n_episodes": n_episodes}
        for k in self.attrs:
            param[k] = getattr(self, k)
        json.dump(param, open(os.path.join(fld, "param.json"), "w"))

    def __sample_db(self):
        prices, title = self.db[self.i_db]
        self.i_db += 1
        if self.i_db == self.n_db:
            self.i_db = 0
        return prices, title


class SinSampler(Sampler):

    def __init__(
        self,
        game,
        window_episode=None,
        noise_amplitude_ratio=None,
        period_range=None,
        amplitude_range=None,
        fld=None,
    ):

        self.n_var = 1  # price only

        self.window_episode = window_episode
        self.noise_amplitude_ratio = noise_amplitude_ratio
        self.period_range = period_range
        self.amplitude_range = amplitude_range
        self.can_half_period = False

        self.attrs = [
            "title",
            "window_episode",
            "noise_amplitude_ratio",
            "period_range",
            "amplitude_range",
            "can_half_period",
        ]

        param_str = str(
            (self.noise_amplitude_ratio, self.period_range, self.amplitude_range)
        )
        if game == "single":
            self.sample = self.__sample_single_sin
            self.title = "SingleSin" + param_str
        elif game == "concat":
            self.sample = self.__sample_concat_sin
            self.title = "ConcatSin" + param_str
        elif game == "concat_half":
            self.can_half_period = True
            self.sample = self.__sample_concat_sin
            self.title = "ConcatHalfSin" + param_str
        elif game == "concat_half_base":
            self.can_half_period = True
            self.sample = self.__sample_concat_sin_w_base
            self.title = "ConcatHalfSin+Base" + param_str
            self.base_period_range = (
                int(2 * self.period_range[1]),
                4 * self.period_range[1],
            )
            self.base_amplitude_range = (20, 80)
        elif game == "load":
            self.load_db(fld)
        else:
            raise ValueError

    def __rand_sin(
        self,
        period_range=None,
        amplitude_range=None,
        noise_amplitude_ratio=None,
        full_episode=False,
    ):

        if period_range is None:
            period_range = self.period_range
        if amplitude_range is None:
            amplitude_range = self.amplitude_range
        if noise_amplitude_ratio is None:
            noise_amplitude_ratio = self.noise_amplitude_ratio

        period = random.randrange(period_range[0], period_range[1])
        amplitude = random.randrange(amplitude_range[0], amplitude_range[1])
        noise = noise_amplitude_ratio * amplitude

        if full_episode:
            length = self.window_episode
        else:
            if self.can_half_period:
                length = int(random.randrange(1, 4) * 0.5 * period)
            else:
                length = period

        p = 100.0 + amplitude * np.sin(np.array(range(length)) * 2 * 3.1416 / period)
        p += np.random.random(p.shape) * noise

        return p, "100+%isin((2pi/%i)t)+%ie" % (amplitude, period, noise)

    def __sample_concat_sin(self):
        prices = []
        p = []
        while True:
            p = np.append(p, self.__rand_sin(full_episode=False)[0])
            if len(p) > self.window_episode:
                break
        prices.append(p[: self.window_episode])
        return np.array(prices).T, "concat sin"

    def __sample_concat_sin_w_base(self):
        prices = []
        p = []
        while True:
            p = np.append(p, self.__rand_sin(full_episode=False)[0])
            if len(p) > self.window_episode:
                break
        base, base_title = self.__rand_sin(
            period_range=self.base_period_range,
            amplitude_range=self.base_amplitude_range,
            noise_amplitude_ratio=0.0,
            full_episode=True,
        )
        prices.append(p[: self.window_episode] + base)
        return np.array(prices).T, "concat sin + base: " + base_title

    def __sample_single_sin(self):
        prices = []
        funcs = []
        p, func = self.__rand_sin(full_episode=True)
        prices.append(p)
        funcs.append(func)
        return np.array(prices).T, str(funcs)


def find_ideal(p, just_once):
    if not just_once:
        diff = np.array(p[1:]) - np.array(p[:-1])
        return sum(np.maximum(np.zeros(diff.shape), diff))
    else:
        best = 0.0
        i0_best = None
        for i in range(len(p) - 1):
            best = max(best, max(p[i + 1 :]) - p[i])

        return best


class Market:
    """
    state 			MA of prices, normalized using values at t
                                    ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
                                    which is self.state_shape

    action 			three action
                                    0:	empty, don't open/close.
                                    1:	open a position
                                    2: 	keep a position
    """

    def reset(self, rand_price=True):
        self.empty = True
        if rand_price:
            prices, self.title = self.sampler.sample()
            price = np.reshape(prices[:, 0], prices.shape[0])

            self.prices = prices.copy()
            self.price = price / price[0] * 100
            self.t_max = len(self.price) - 1

            # breakpoint()

        self.max_profit = find_ideal(self.price[self.t0 :], False)
        self.t = self.t0
        return self.get_state(), self.get_valid_actions()

    def get_state(self, t=None):
        if t is None:
            t = self.t
        state = self.prices[t - self.window_state + 1 : t + 1, :].copy()
        for i in range(self.sampler.n_var):
            norm = np.mean(state[:, i])
            state[:, i] = (state[:, i] / norm - 1.0) * 100
        return state

    def get_valid_actions(self):
        if self.empty:
            return [0, 1]  # wait, open
        else:
            return [0, 2]  # close, keep

    def get_noncash_reward(self, t=None, empty=None):
        if t is None:
            t = self.t
        if empty is None:
            empty = self.empty
        reward = self.direction * (self.price[t + 1] - self.price[t])
        if empty:
            reward -= self.open_cost
        if reward < 0:
            reward *= 1.0 + self.risk_averse
        return reward

    def step(self, action):

        done = False
        if action == 0:  # wait/close
            reward = 0.0
            self.empty = True
        elif action == 1:  # open
            reward = self.get_noncash_reward()
            self.empty = False
        elif action == 2:  # keep
            reward = self.get_noncash_reward()
        else:
            raise ValueError("no such action: " + str(action))

        self.t += 1
        return self.get_state(), reward, self.t == self.t_max, self.get_valid_actions()

    def __init__(
        self, sampler, window_state, open_cost, direction=1.0, risk_averse=0.0
    ):

        self.sampler = sampler
        self.window_state = window_state
        self.open_cost = open_cost
        self.direction = direction
        self.risk_averse = risk_averse

        self.n_action = 3
        self.state_shape = (window_state, self.sampler.n_var)
        self.action_labels = ["empty", "open", "keep"]
        self.t0 = window_state - 1


class MarketSimulator:
    def __init__(self, fld=None, window_state: int = 40, open_cost: float = 3.3):

        # if fld is None:

        sampler = SinSampler(
            "load",
            fld="/home/vinybrasil/projects/env_trading/data/SinSamplerDB/concat_half_base_B",
        )
        self.env = Market(sampler, window_state, open_cost)
        # env_t = 0
        # breakpoint()
        # simulator = SimpleSimulator(env)

        # self.env = env


# if __name__ == '__main__':
#     env2 = MarketSimulator()
#     state, valid_actions = env2.env.reset()

# faz o treino; testa; faz o treino; testa; ...
# breakpoint()
