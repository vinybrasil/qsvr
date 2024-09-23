import random
import uuid
from datetime import datetime

import numpy as np

from qsvr.enviroments.shortest_path import ShortestPath
from qsvr.helpers import log_ct_estimates, log_ct_summary, play_sp

SEED = 424242
np.random.seed(SEED)


discount_factor = 0.99
eps = 0.99
eps_decay_factor = 0.99
max_steps = 150
learning_rate = 0.1
num_episodes = 200

cost_matrix = np.array(
    [
        [0, 4, 0, 0, 0, 0, 0, 8, 0],
        [4, 0, 8, 0, 0, 0, 0, 11, 0],
        [0, 8, 0, 7, 0, 4, 0, 0, 3],
        [0, 0, 7, 0, 9, 14, 0, 0, 0],
        [0, 0, 0, 9, 0, 10, 0, 0, 0],
        [0, 0, 4, 14, 10, 0, 3, 0, 0],
        [0, 0, 0, 0, 0, 3, 0, 3, 4],
        [8, 11, 0, 0, 0, 0, 3, 0, 5],
        [0, 0, 3, 0, 0, 0, 4, 5, 0],
    ], dtype=float
)

# env = ShortestPath(cost_matrix, source=4, target=8)  # 3, 0
env = ShortestPath(cost_matrix, source=3, target=0)
#q_table = np.zeros([env.observation_space, env.action_space])

q_table = np.random.uniform(-0.1, 0.1, size=(env.observation_space, env.action_space))

estimates = []
estimates.append(["episode", "state", "action", "q_s_a"])

summary = []
summary.append(
    ["episode", "steps", "total_reward", "optimal_policy", "random_percentage"]
)


model_id = str(uuid.uuid4())
curr_dt = int(round(datetime.now().timestamp()))


for episode in range(num_episodes):
    state = env.reset()
    done = False
    #print(f"EPISODE {episode}----------------------")
    eps *= eps_decay_factor
    steps = 0
    total_reward = 0
    random_choice_counter = 0

    if episode > 0:
        for s in range(env.observation_space):
            for a in range(env.action_space):
                estimates.append([episode, s, a, q_table[s][a]])

    while steps < max_steps:

        valid_actions = env.get_valid_actions()
        if np.random.random() < eps:

            action = np.random.choice(valid_actions)
            choice = "random"
            random_choice_counter += 1

        else:

            q_of_next_states = q_table[state][valid_actions]
            action = valid_actions[np.argmax(q_of_next_states)]
            choice = "model"

        new_state, done, reward = env.step(action)

        total_reward += reward

        #print(state, action, new_state, reward, done, choice)

        # q_table[state, action] = (1 - learning_rate) * q_table[state, action]
        # + learning_rate * (reward + discount_factor * np.max(q_table[new_state]))
        # state = new_state

        #breakpoint()

        valid_actions = env.get_valid_actions(new_state)

        q_table[state, action] = (1 - learning_rate) * q_table[state, action]
        + learning_rate * (reward + discount_factor * np.max(q_table[new_state][valid_actions]))

        state = new_state

        steps += 1

        if done == True:
            break

    summary.append(
        [
            episode,
            steps,
            total_reward,
            play_sp(q_table, env, False),
            round(random_choice_counter / steps, 2),
        ]
    )


#log_ct_estimates(estimates, model_id, curr_dt, model_type="qlearning", enviroment="sp")
#log_ct_summary(summary, model_id, curr_dt, "qlearning", "sp")


# breakpoint()

# play_sp(q_table, env)

# breakpoint()
