import random
import uuid
from datetime import datetime

import numpy as np

from qsvr.enviroments.shortest_path import ShortestPath
from qsvr.helpers import log_ct_estimates, log_ct_summary, play_sp
from qsvr.onlinesvr import OnlineSVR

discount_factor = 0.99  # 0.95, 0.99
eps = 0.99
eps_decay_factor = 0.99  # 0.9
max_steps = 150
num_episodes = 150

UPDATE_FREQUENCY = 1  # 10
log_interval = 149
HULL = False

SEED = 23
np.random.seed(SEED)

estimates = []
estimates.append(["episode", "state", "action", "q_s_a"])

summary = []
summary.append(
    ["episode", "steps", "total_reward", "optimal_policy", "random_percentage"]
)

model_id = str(uuid.uuid4())
curr_dt = int(round(datetime.now().timestamp()))


model = OnlineSVR(
    # numFeatures=5,
    numFeatures=2,
    C=1.0,  # 1
    eps=0.1,  # 0.000001
    kernelParam=1.0,  # 0.7, G: 0.3, 0.7; R: 40, 30, 100, 0.7
    kerneltype=b"G",  # G, R
    bias=0,
    debug=False,
    debug_time=False,
    engine="cpp",
    # engine="python",
)

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
    ]
)

env = ShortestPath(cost_matrix, source=3, target=0)
# env = ShortestPath(cost_matrix, source=3, target=0)


def build_q_table(model):
    y_preds = [
        [
            float(
                model.predict(
                    np.append(
                        i,
                        a,
                    )
                )
            )
            for i in range(env.observation_space)
        ]
        for a in range(env.action_space)
    ]
    q_table = np.zeros([env.observation_space, env.action_space])
    for k in range(env.action_space):
        q_table[:, k] = y_preds[k]
    return q_table


for episode in range(num_episodes):
    state = env.reset()
    done = False
    print(f"EPISODE {episode}----------------------")
    eps *= eps_decay_factor

    steps = 0
    total_reward = 0
    random_choice_counter = 0
    if episode > 0:
        for s1 in range(env.observation_space):
            for a1 in range(env.action_space):
                estimates.append(
                    [episode, s1, a1, float(model.predict(np.append(s1, a1)))]
                )

    while steps < max_steps:

        valid_actions = env.get_valid_actions()

        if np.random.random() < eps:

            action = np.random.choice(valid_actions)
            random_choice_counter += 1

            choice = 'random'
        else:

            actions = [float(model.predict(np.append(state, i))) for i in valid_actions]

            action = valid_actions[np.argmax(actions)]

            choice = 'model'
            #pass

        new_state, done, reward = env.step(action)

        total_reward += reward
        print(state, new_state, action, reward, done, choice)

        try:
            all_actions_values = [
                float(model.predict(np.append(new_state, i))) for i in valid_actions
            ]

            a_1 = np.argmax(np.array(all_actions_values))
            q_1 = float(model.predict(np.append(new_state, a_1)))

            target = reward + (discount_factor * q_1)

            if done == True:
                target = reward

        except:
            print(
                "WARNING: model not available to create the TD error. Using just the reward as target."
            )
            target = reward

        model.learn(
            np.append(state, action),
            target + np.random.normal(0, 0.01),
        )

        state = new_state

        if done == True:
            break

        steps += 1

    q_table = build_q_table(model)
    summary.append(
        [
            episode,
            steps,
            total_reward,
            play_sp(q_table, env, True),
            round(random_choice_counter / steps, 2),
        ]
    )


q_table = build_q_table(model)


#log_ct_estimates(estimates, model_id, curr_dt, model_type="qsvr", enviroment="sp")
#log_ct_summary(summary, model_id, curr_dt, "qsvr", "sp")


breakpoint()

#play_sp(q_table, env)

#breakpoint()
