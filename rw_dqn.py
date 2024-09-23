import random
import uuid
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from qsvr.enviroments.random_walk import RandomWalk
from qsvr.helpers import log_ct_estimates, log_ct_summary, play_rw


SEED = 222
np.random.seed(SEED)


num_states = 15
n_time_step = num_states * 5
env = RandomWalk(num_states, max_steps=n_time_step)
# env.set_seed(SEED)


EPSILON2 = 0.99
EPSILON2_DECAY = 0.99  # 0.99 e 0.99 dá parecido
TARGET_UPDATE_FREQUENCY = 5  # 10
num_episodes = 200

GAMMA = 0.99  # 0.99
LEARNING_RATE = 1e-5  # 1e-3
MEMORY_SIZE = 200  # 1000
BATCH_SIZE = 32  # 64

summary = []
summary.append(
    ["episode", "steps", "total_reward", "optimal_policy", "random_percentage"]
)
random_choice_counter = 0

model_id = str(uuid.uuid4())
curr_dt = int(round(datetime.now().timestamp()))


class ReplayMemory:
    def __init__(self, n_s, n_a, memory_size, batch_size):
        self.n_s = n_s
        self.n_a = n_a

        self.MEMORY_SIZE = memory_size
        self.BATCH_SIZE = batch_size  # 64
        self.all_s = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        self.all_a = np.random.randint(
            low=0, high=self.n_a, size=self.MEMORY_SIZE, dtype=np.uint8
        )
        self.all_r = np.empty(self.MEMORY_SIZE, dtype=np.float64)
        self.all_done = np.random.randint(
            low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8
        )
        self.all_s_ = np.empty(shape=(self.MEMORY_SIZE, self.n_s), dtype=np.float64)
        self.count = 0
        self.t = 0

        # self.a1 = np.random.randint(low=0,high=)

    def add_memo(self, s, a, r, done, s_):
        self.all_s[self.t] = s
        self.all_a[self.t] = a
        self.all_r[self.t] = r
        self.all_done[self.t] = done
        self.all_s_[self.t] = s_
        self.count = max(self.count, self.t + 1)
        self.t = (self.t + 1) % self.MEMORY_SIZE

    def sample(self):
        if self.count < self.BATCH_SIZE:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0, self.count), self.BATCH_SIZE)

        batch_s = []
        batch_a = []
        batch_r = []
        batch_done = []
        batch_s_ = []
        for idx in indexes:
            batch_s.append(self.all_s[idx])
            batch_a.append(self.all_a[idx])
            batch_r.append(self.all_r[idx])
            batch_done.append(self.all_done[idx])
            batch_s_.append(self.all_s_[idx])

        batch_s_tensor = torch.as_tensor(np.asarray(batch_s), dtype=torch.float32)
        batch_a_tensor = torch.as_tensor(
            np.asarray(batch_a), dtype=torch.int64
        ).unsqueeze(-1)
        batch_r_tensor = torch.as_tensor(
            np.asarray(batch_r), dtype=torch.float32
        ).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(
            np.asarray(batch_done), dtype=torch.float32
        ).unsqueeze(-1)
        batch_s__tensor = torch.as_tensor(np.asarray(batch_s_), dtype=torch.float32)

        return (
            batch_s_tensor,
            batch_a_tensor,
            batch_r_tensor,
            batch_done_tensor,
            batch_s__tensor,
        )


class DQN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()  # Reuse the param of nn.Module
        in_features = n_input  # ?

        # nn.Sequential() ?
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tanh(), nn.Linear(64, n_output)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs, raw_q=False):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_tensor.unsqueeze(0))  # ?
        # max_q_index = torch.argmax(q_values, dim=1)[0]  # ?
        if raw_q:
            return q_values.detach().numpy()

        max_q_index = torch.argmax(q_values)
        action = max_q_index.detach().item()  # get the idx of q
        return action


class Agent:
    def __init__(
        self,
        idx,
        n_input,
        n_output,
        gamma,
        learning_rate,
        memory_size,
        batch_size,
        mode="train",
    ):
        self.idx = idx
        self.mode = mode
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = gamma
        self.learning_rate = learning_rate
        # self.MIN_REPLAY_SIZE = 1000

        self.memo = ReplayMemory(
            n_s=self.n_input,
            n_a=self.n_output,
            memory_size=memory_size,
            batch_size=batch_size,
        )

        # Initialize the replay buffer of agent i
        if self.mode == "train":
            self.online_net = DQN(self.n_input, self.n_output)
            self.target_net = DQN(self.n_input, self.n_output)

            self.target_net.load_state_dict(
                self.online_net.state_dict()
            )  # copy the current state of online_net

            self.optimizer = torch.optim.Adam(
                self.online_net.parameters(), lr=self.learning_rate
            )


def build_qtable(agent):

    # s = env.reset()

    q_values = []
    for s in range(num_states):
        a = agent.online_net.act(s, raw_q=True)
        q_values.append(a)

    return np.array(q_values)




n_state = 1
n_action = env.action_space

agent = Agent(
    idx=0,
    n_input=n_state,
    n_output=n_action,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    memory_size=MEMORY_SIZE,
    batch_size=BATCH_SIZE,
    mode="train",
)

transitions = []

estimates = []
estimates.append(["episode", "state", "action", "q_s_a"])

REWARD_BUFFER = np.zeros(shape=num_episodes)
for episode in range(num_episodes):
    s = env.reset()
    done = False
    print(f"EPISODE {episode}----------------------")

    # print("OPTIMAL POLICY: ", build_qtable(agent))
    episode_reward = 0
    EPSILON2 *= EPSILON2_DECAY

    if episode > 0:
        for state in range(env.observation_space):
            q_values = agent.online_net.act(state, raw_q=True)
            for a in range(env.action_space):
                estimates.append([episode, state, a, q_values[a]])

    for episode_i in range(n_time_step):

        epsilon = EPSILON2
        # if not isinstance(state, int):
        #     state = state[0]

        random_sample = random.random()
        if random_sample <= epsilon:
            a = np.random.randint(0, env.action_space)
            choice = "random"
        else:
            a = agent.online_net.act(s)
            choice = "model"

        s_, r, done = env.step(a)
        transitions.append([s, s_, a, r, done, choice])

        print(s, s_, a, r, done, choice)
        # total_reward += reward
        agent.memo.add_memo(s, a, r, done, s_)
        s = s_
        episode_reward += r

        if done:
            # print(step_i,done)
            s = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        # Start Gradient Step
        batch_s, batch_a, batch_r, batch_done, batch_s_ = (
            agent.memo.sample()
        )  # update batch-size amounts of Q

        # Compute Targets
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # ?
        targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

        # Compute Q_values
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)  # ?

        # Compute Loss
        loss = nn.functional.smooth_l1_loss(a_q_values, targets)

        # Gradient Descent
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        # Print the training progress
        print("Episode: {}".format(episode_i))
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))

    if episode > 0:
        q_table = build_qtable(agent)
        summary.append(
            [
                episode,
                episode_i + 1,
                episode_reward,
                play_rw(q_table, env, False),
                round(random_choice_counter / episode_i, 2),
            ]
        )


log_ct_summary(summary, model_id, curr_dt, "dqn", "rw", str(num_states))
log_ct_estimates(estimates, model_id, curr_dt, model_type="dqn", enviroment="rw", number_states=str(num_states))

play_rw(q_table, env, False)

#breakpoint()
