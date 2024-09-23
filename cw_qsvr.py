import csv
import uuid
from datetime import datetime

import numpy as np

# import gymnasium as gym
from qsvr.enviroments.chain_walk import ChainWalk
from qsvr.helpers import (log_ct_estimates, log_ct_summary, log_ct_transitions,
                          play_cw)
from qsvr.onlinesvr import OnlineSVR

SEED = 222
np.random.seed(SEED)

env = ChainWalk()
env.set_seed(SEED)

discount_factor = 0.99
eps = 0.99
eps_decay_factor = 0.99
num_episodes = 200
max_steps = 4

model = OnlineSVR(
    numFeatures=2,
    C=1.0,
    eps=0.1,
    kernelParam=0.7,
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


transitions = []
summary = []
summary.append(
    ["episode", "steps", "total_reward", "optimal_policy", "random_percentage"]
)

transitions.append(
    ["state", "new_state", "action", "reward", "done", "choice", "q_s_a"]
)

estimates = []
estimates.append(["episode", "state", "action", "q_s_a"])


model_id = str(uuid.uuid4())
curr_dt = int(round(datetime.now().timestamp()))

for episode in range(num_episodes):
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    print(f"EPISODE {episode}----------------------")

    steps = 0
    random_choice_counter = 0
    # if episode % 50 == 0:
    #     breakpoint()

    total_reward = 0
    if episode > 0:
        for s in range(env.observation_space):
            for a in range(env.action_space):
                estimates.append([episode, s, a, float(model.predict(np.append(s, a)))])

    while steps < max_steps:

        if not isinstance(state, int):
            state = state[0]

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
            # breakpoint()
            action = np.argmax(np.array(actions))
            choice = "model"

        new_state, reward, done = env.step(action)

        total_reward += reward
        print(state, new_state, action, reward, done )

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
            # print(all_actions_values )

            a_1 = np.argmax(np.array(all_actions_values))
            # print(a_1)
            q_1 = float(model.predict(np.append(new_state, a_1)))
            # print(q_1)

            # print(f'{a_1}, {q_1}')
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

        if episode > 0:
            transitions.append(
                [state, new_state, action, reward, done, choice, actions]
            )

        model.learn(
            # np.append(np.identity(env.observation_space)[state : state + 1], action),
            np.append(state, action),
            target,
        )
        state = new_state

        steps += 1

        if done == True:
            break

    if episode > 0:
        q_table = build_qtable(model)
        print("OPTIMAL POLICY: ", play_cw(q_table))

        # summary.append(['episode', 'steps', 'total_reward', 'optimal_policy', 'random_percentage'])
        summary.append(
            [
                episode,
                steps,
                total_reward,
                list(play_cw(q_table)),
                round(random_choice_counter / steps, 2),
            ]
        )

    print("total_reward: ", total_reward)
# env.close()

# #log_ct_transitions(transitions, model_id, curr_dt, model_type="qsvr", enviroment="cw")


log_ct_summary(summary, model_id, curr_dt, "qsvr", "cw")
log_ct_estimates(estimates, model_id, curr_dt, model_type="qsvr", enviroment="cw")


# with open("result.csv", mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerows(transitions)


breakpoint()
