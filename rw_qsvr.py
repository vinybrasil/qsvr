import csv
import uuid
from datetime import datetime

import numpy as np

from qsvr.enviroments.random_walk import RandomWalk
from qsvr.helpers import log_ct_estimates, log_ct_summary, play_rw
from qsvr.onlinesvr import OnlineSVR

# env = gym.make("FrozenLake-v1", is_slippery=False)




discount_factor = 0.99
eps = 0.99
eps_decay_factor = 0.99
num_episodes = 150

num_states = 15
max_steps = num_states * 5

env = RandomWalk(num_states, max_steps=max_steps)

SEED = 194
np.random.seed(SEED)


model = OnlineSVR(
    numFeatures=2,
    C=1.0,
    eps=0.1,
    kernelParam=0.4,
    kerneltype=b"G",
    bias=0,
    debug=False,
)


def build_qtable(model):
    y_preds = [
        [float(model.predict(np.append(v, a))) for v in range(env.observation_space)]
        for a in range(env.action_space)
    ]
    q_table = np.zeros([env.observation_space, env.action_space])
    for k in range(env.action_space):
        q_table[:, k] = y_preds[k]

    return q_table


summary = []
summary.append(
    ["episode", "steps", "total_reward", "optimal_policy", "random_percentage"]
)

model_id = str(uuid.uuid4())
curr_dt = int(round(datetime.now().timestamp()))

estimates = []
estimates.append(["episode", "state", "action", "q_s_a"])

for episode in range(num_episodes):
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    random_choice_counter = 0
    steps = 0
    print(f"EPISODE {episode}----------------------")

    total_reward = 0

    if episode > 0:
        for s in range(env.observation_space):
            for a in range(env.action_space):
                estimates.append([episode, s, a, float(model.predict(np.append(s, a)))])

    while steps < max_steps:

        if np.random.random() < eps:
            action = np.random.randint(0, env.action_space)
            choice = "random"
            random_choice_counter += 1

        else:

            actions = [
                float(
                    model.predict(
                        np.append(
                            state,
                            k,
                        )
                    )
                )
                for k in range(env.action_space)
            ]
            action = np.argmax(np.array(actions))
            choice = "model"

        new_state, reward, done = env.step(action)
        print(state, action, new_state, reward, done, choice)

        total_reward += reward

        try:
            actions = [
                float(
                    model.predict(
                        np.append(
                            new_state,
                            q,
                        )
                    )
                )
                for q in range(env.action_space)
            ]

            all_actions_values = [
                float(model.predict(np.append(new_state, i)))
                for i in range(env.action_space)
            ]

            a_1 = np.argmax(np.array(all_actions_values))
            q_1 = float(model.predict(np.append(new_state, a_1)))
            target = (
                reward
                + (discount_factor * q_1)
                # - float(model.predict(np.append(state, action)))
            )

            if done == True:
                target = reward

        except:
            print("DEU RUIM")
            target = reward

        model.learn(
            np.append(state, action),
            target,
        )
        state = new_state

        steps += 1
        if done == True:
            break

        steps += 1
    if episode > 0:
        q_table = build_qtable(model)
        #print("OPTIMAL POLICY: ", play_rw(q_table, env))
        print("--")
        summary.append(
            [
                episode,
                steps,
                total_reward,
                list(play_rw(q_table, env, write=True)),
                round(random_choice_counter / steps, 2),
            ]
        )



# log_ct_summary(summary, model_id, curr_dt, "qsvr", "rw", str(num_states))
# log_ct_estimates(estimates, model_id, curr_dt, model_type="qsvr", enviroment="rw", number_states=str(num_states))

breakpoint()
