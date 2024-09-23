import csv

import dill
import gymnasium as gym
import numpy as np

from .onlinesvr import MatrixClass


def save_model(model, model_id, timestamp, env="ct"):
    model.matrixclass_instance = None
    with open(f"models/{env}/qsvr/{timestamp}_{model_id}.pkl", "wb") as f:
        dill.dump(model, f)
    model.matrixclass_instance = MatrixClass()


def load_model(model_name, env="ct"):
    with open(f"models/{env}/qsvr/{model_name}.pkl", "rb") as f:
        model = dill.load(f)
    model.matrixclass_instance = MatrixClass()
    return model


# Discretize state function
def discretize_state(state, num_discretized_spaces):
    # breakpoint()
    state_space_size = [num_discretized_spaces for i in range(4)]

    # state_bounds = [env.observation_space.low, env.observation_space.high]
    state_bounds = [
        np.array([-4.8000002e00, -3.4028235e38, -4.1887903e-01, -3.4028235e38]),
        np.array([4.8000002e00, 3.4028235e38, 4.1887903e-01, 3.4028235e38]),
    ]

    state_bounds[0][1] = -2.4
    state_bounds[0][3] = -2.4

    state_bounds[1][1] = 2.4
    state_bounds[1][3] = 2.4

    state_discretize = [
        np.linspace(
            state_bounds[0][i],  # state_bounds[0][i],
            state_bounds[1][i],  # state_bounds[1][i],
            state_space_size[i],
        )
        for i in range(4)
    ]

    discretized_state = []
    for i in range(4):
        discretized_state.append(np.digitize(state[i], state_discretize[i]) - 1)

    # breakpoint()
    return discretized_state


def play_ct(
    model,
    model_type="qsvr",
    render_mode="human",
    num_discretized_spaces=None,
    debug=True,
):
    env = gym.make(id="CartPole-v1", render_mode=render_mode)

    obs, _ = env.reset()
    # obs = obs[[0, 2]]
    j = 0
    k = 0
    list_num_times = []
    while j < 10:
        k += 1
        if model_type == "qsvr":
            if num_discretized_spaces is not None:
                obs = discretize_state(obs, num_discretized_spaces)
            actions = [
                float(model.predict(np.append(obs, i)))
                for i in range(env.action_space.n)
            ]
            a = np.argmax(np.array(actions))

        if model_type == "qlearning":
            # aqui o modelo é uma table

            obs = discretize_state(obs, num_discretized_spaces)

            a = np.argmax(model[obs[0], obs[1], obs[2], obs[3], :])

        if model_type == "dqn":
            a = model.online_net.act(obs)

        obs, r, done, info, _ = env.step(a)
        # obs = obs[[0, 2]]

        if done or (k > 3001):
            list_num_times.append(k)
            if debug:
                print(f"num_times: {k}")

            
            obs, _ = env.reset()
            # obs = obs[[0, 2]]

            j += 1
            k = 0
    media = np.array(list_num_times).mean()
    std = np.array(list_num_times).std()
    print(f"MEAN: {media}, STD: {std}")
    env.close()
    env = gym.make(id="CartPole-v1", render_mode=None)
    return (list_num_times, media, std)


def log_ct_transitions(
    transitions, model_id, timestamp, model_type="qsvr", enviroment="ct"
):
    with open(
        f"resultados/{enviroment}/{model_type}/transitions/{timestamp}_{model_id}.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(transitions)


def log_ct_summary(summary, model_id, timestamp, model_type="qsvr", enviroment="ct", number_states=None):
    if number_states is None:
        with open(
            f"resultados/{enviroment}/{model_type}/summary/{timestamp}_{model_id}.csv",
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerows(summary)
        
        return 
    
    with open(
        f"resultados/{enviroment}/{model_type}/summary/{number_states}/{timestamp}_{model_id}.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(summary)



def log_ct_estimates(
    estimates, model_id, timestamp, model_type="qsvr", enviroment="ct", number_states=None
):
    if number_states is None:
        with open(
            f"resultados/{enviroment}/{model_type}/estimates/{timestamp}_{model_id}.csv",
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerows(estimates)

        return 
    
    with open(
        f"resultados/{enviroment}/{model_type}/estimates/{number_states}/{timestamp}_{model_id}.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(estimates)


def log_ct_parameters(
    parameters, model_id, timestamp, model_type="qsvr", enviroment="ct"
):
    with open(
        f"resultados/{enviroment}/{model_type}/parameters/{timestamp}_{model_id}.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(parameters)


def mov_new_position(move, position):
    # print(position)
    table = np.zeros([4, 4])

    table[int(np.floor((position) / 4))][(position % 4)] = 1

    # print(table)
    if move == 0:
        if position not in [0, 4, 8, 12]:
            table[int(np.floor((position) / 4))][(position % 4)] = 0
            position -= 1
            table[int(np.floor((position) / 4))][(position % 4)] = 1

    if move == 1:
        if position not in [12, 13, 14, 15]:
            table[int(np.floor((position) / 4))][(position % 4)] = 0
            position += 4
            table[int(np.floor((position) / 4))][(position % 4)] = 1

    if move == 2:
        if position not in [3, 7, 11, 15]:
            table[int(np.floor((position) / 4))][(position % 4)] = 0
            position += 1
            table[int(np.floor((position) / 4))][(position % 4)] = 1

    if move == 3:
        if position not in [0, 1, 2, 3]:
            table[int(np.floor((position) / 4))][(position % 4)] = 0
            position -= 4
            table[int(np.floor((position) / 4))][(position % 4)] = 1

    # print(table)
    return position


def check_dead(position: int) -> bool:
    if position in [5, 7, 11, 12]:
        return True
    return False


def play_fl(Q, starting_position=0):
    positions = []
    position = starting_position
    positions.append(position)
    end = False
    i = 0
    while end != True:
        print("Position: ", position)
        if check_dead(position):
            print("dead")
            end = True
            break
        if position == 15:
            print("finished game")
            end = True
            break
        move = np.argmax(Q, axis=1)[position]
        position = mov_new_position(move, position)
        positions.append(position)
        print("Move: ", move)
        i += 1
        if i > 50:
            break
    table = np.zeros([4, 4])
    for position in positions:

        table[int(np.floor((position) / 4))][(position % 4)] = 1
        print(table)
        table[int(np.floor((position) / 4))][(position % 4)] = 0
    return


def play_cw(q_table):
    return np.argmax(q_table, axis=1)


def play_sp(q_table, env, write=True):
    state = env.reset()
    path = []
    for i in range(9):
        path.append(state)
        valid_actions = env.get_valid_actions()

        q_of_next_states = q_table[state][valid_actions]
        action = valid_actions[np.argmax(q_of_next_states)]

        new_state, done, reward = env.step(action)
        if write:
            print(state, action, new_state, reward, done)

        if done == True:
            path.append(new_state)
            break

        state = new_state

    # path.append(state)

    return path


def play_rw(q_table, env, write=True):
    state = env.reset()
    path = []
    for i in range(env.observation_space):
        path.append(state)

        q_of_next_states = q_table[state]
        action = np.argmax(q_of_next_states)

        new_state, reward, done = env.step(action)

        if write:
            print(state, action, new_state, reward, done)

        if done == True:
            path.append(new_state)
            break

        state = new_state

    return path
