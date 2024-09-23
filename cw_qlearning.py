import uuid
from datetime import datetime

import numpy as np

from qsvr.enviroments.chain_walk import ChainWalk
from qsvr.helpers import log_ct_estimates, log_ct_summary, play_cw

discount_factor = 0.99
eps = 0.99
eps_decay_factor = 0.99
learning_rate = 0.1
num_episodes = 200
max_steps = 4

SEED = 222
np.random.seed(SEED)

env = ChainWalk()
env.set_seed(SEED)

#q_table = np.zeros([env.observation_space, env.action_space])
q_table = np.random.uniform(-0.1, 0.1, size=(env.observation_space, env.action_space))#
summary = []
summary.append(
    ["episode", "steps", "total_reward", "optimal_policy", "random_percentage"]
)

estimates = []
estimates.append(["episode", "state", "action", "q_s_a"])


model_id = str(uuid.uuid4())
curr_dt = int(round(datetime.now().timestamp()))


for episode in range(num_episodes):
    state = env.reset()
    # print(env.render('ansi'))
    eps *= eps_decay_factor
    done = False
    total_reward = 0
    steps = 0
    random_choice_counter = 0

    if episode > 0:
        for s in range(env.observation_space):
            for a in range(env.action_space):
                estimates.append([episode, s, a, q_table[s][a]])

    print(f"EPISODE {episode}----------------------")

    # if episode > 1:

    while steps < max_steps:

        if np.random.random() < eps:
            action = np.random.randint(0, 2)
            random_choice_counter += 1
        else:
            action = np.argmax(q_table[state, :])

        new_state, reward, done = env.step(action)

        total_reward += reward

        print(state, new_state, action, reward, done )
        # except:
        #    breakpoint()

        # delta_q = learning_rate * ( reward +
        #     discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        # )

        # q_table[state, action] += delta_q

        q_table[state, action] = (1 - learning_rate) * q_table[
            state, action
        ] + learning_rate * (
            reward
            + discount_factor * np.max(q_table[new_state, :])
            - q_table[state, action]
        )

        state = new_state

        steps += 1

        if done == True:
            break

        # print("reward:", reward)
        # print("Done?", done)
        # print(env.render('ansi'))
        # input("")
    # env.close()
    # if episode > 1:
    #     q_table = build_qtable(model)
    #     print("OPTIMAL POLICY: ", play_cw(q_table))

    # summary.append(['episode', 'steps', 'total_reward', 'optimal_policy', 'random_percentage'])

    print("OPTIMAL POLICY: ", play_cw(q_table))
    summary.append(
        [
            episode,
            steps,
            total_reward,
            list(play_cw(q_table)),
            round(random_choice_counter / steps, 2),
        ]
    )

log_ct_summary(summary, model_id, curr_dt, "qlearning", "cw")
log_ct_estimates(estimates, model_id, curr_dt, model_type="qlearning", enviroment="cw")

breakpoint()
