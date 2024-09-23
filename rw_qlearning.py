import random
import uuid
from datetime import datetime

import numpy as np

from qsvr.enviroments.random_walk import RandomWalk
from qsvr.helpers import log_ct_estimates, log_ct_summary, play_rw

# from helpers import play_game

# env = gym.make("CartPole-v1")

discount_factor = 0.99
eps = 0.99
eps_decay_factor = 0.99
learning_rate = 0.1
num_episodes = 200
num_states = 15
max_steps = num_states * 5

SEED = 222
np.random.seed(SEED)

env = RandomWalk(num_states, max_steps=max_steps)

summary = []
summary.append(
    ["episode", "steps", "total_reward", "optimal_policy", "random_percentage"]
)


model_id = str(uuid.uuid4())
curr_dt = int(round(datetime.now().timestamp()))

# q_table = np.zeros([env.observation_space, env.action_space])
q_table = np.random.uniform(-1, 1, size=(env.observation_space, env.action_space))

estimates = []
estimates.append(["episode", "state", "action", "q_s_a"])

#breakpoint()

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    random_choice_counter = 0
    steps = 0
    eps *= eps_decay_factor
    print(f"EPISODE {episode}----------------------")

    if episode > 0:
        for s in range(env.observation_space):
            for a in range(env.action_space):
                estimates.append([episode, s, a, q_table[s][a]])

    while steps < max_steps:

        if np.random.random() < eps:
            action = np.random.randint(0, 2)
            random_choice_counter += 1
        else:
            action = np.argmax(q_table[state - 1, :])

        new_state, reward, done = env.step(action)

        # print(state, action, new_state, reward, done)
        # delta_q =  learning_rate * (
        #     reward +
        #     discount_factor * np.max(q_table[new_state - 1, :])
        #     - q_table[state - 1, action]
        # )

        # q_table[state - 1, action] += delta_q

        q_table[state, action] = (1 - learning_rate) * q_table[
            state - 1, action
        ] + learning_rate * (
            reward
            + discount_factor * np.max(q_table[new_state - 1, :])
            - q_table[state - 1, action]
        )
        state = new_state

        total_reward += reward
        steps += 1

        if done == True:
            break

    #play_rw(q_table, env)
    random_percentage = random_choice_counter / steps
    summary.append([episode, steps, total_reward, play_rw(q_table, env), random_percentage])

# print(*summary, sep="\n")

log_ct_summary(summary, model_id, curr_dt, "qlearning", "rw", str(num_states))

log_ct_estimates(estimates, model_id, curr_dt, model_type="qlearning", enviroment="rw", number_states=str(num_states))

breakpoint()
play_rw(q_table, env)

# summary = []
# summary.append(['episode', 'steps', 'total_reward', 'optimal_policy', 'random_percentage'])
# random_choice_counter = 0

# model_id = str(uuid.uuid4())
# curr_dt = int(round(datetime.now().timestamp()))


# for episode in range(num_episodes):
#     state = env.reset()
#     eps *= eps_decay_factor
#     done = False
#     total_reward = 0

#     print(f"EPISODE {episode}----------------------")


#     while done == False:

#         if np.random.random() < eps:
#             action = np.random.randint(0, 2)
#         else:
#             action = np.argmax(q_table[state, :])

#         new_state, reward, done = env.step(action)

#         total_reward += reward

#         delta_q = reward + learning_rate * (
#             discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
#         )

#         q_table[state, action] += delta_q

#         state = new_state

#     print("OPTIMAL POLICY: ", play_cw(q_table))
#     summary.append([episode, total_reward, str(play_cw(q_table)), 'random_percentage'])

# log_ct_summary(summary, model_id, curr_dt, 'qlearning', 'cw')


# breakpoint()
