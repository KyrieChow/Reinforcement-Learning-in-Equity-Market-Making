import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from buildingblocks.environment import DQNEnv
from rl_agent import RLAgent
from utils.calibration import sample_ref_size_curve, fit_ref_curve, fit_best_volume, sample_ref_spread


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
UPDATE_EVERY = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DNN(nn.Module):
    def __init__(self, n_state, n_action, seed):
        super(DNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(n_state, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_action)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)


class Buffer:
    def __init__(self, n_action, buffer_size, batch_size, seed):
        self.n_action = n_action
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("experience",
                                     field_names=["state", "action", "reward", "state_next", "done"])

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, state_next, done):
        e = self.experience(state, action, reward, state_next, done)
        self.memory.append(e)

    def sampling(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(device)
        states_next = torch.from_numpy(np.vstack(
            [e.state_next for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, states_next, dones


class DQNAgent(RLAgent):
    def __init__(self, n_state, n_action, discount_rate, tau, learning_rate, seed, name: str = None):
        super().__init__(name=name, n_action=n_action)
        self.n_state = n_state
        self.discount_rate = discount_rate
        self.tau = tau
        self.learning_rate = learning_rate
        self.seed = random.seed(seed)

        self.model = DNN(n_state, n_action, seed).to(device)
        self.model_target = DNN(n_state, n_action, seed).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.memory = Buffer(n_action, BUFFER_SIZE, BATCH_SIZE, seed)
        self.counter = 0

    def load_model(model):
        self.model = model

    def step(self, state, action, reward, state_next, done):
        self.memory.add(state, action, reward, state_next, done)

        self.counter = (self.counter + 1) % UPDATE_EVERY
        if self.counter == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sampling()
                self.learn(experiences, self.discount_rate)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            actions = self.model(state)
        self.model.train()

        # eps-greedy
        if random.random() > eps:
            return np.argmax(actions.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.n_action))

    def soft_update(self, model, model_target, tau):
        for param_target, param in zip(model_target.parameters(), model.parameters()):
            param_target.data.copy_(tau * param.data + (1.0 - tau) * param_target.data)

    def learn(self, experiences, discount_rate):
        states, actions, rewards, states_next, dones = experiences
        q_targets_next = self.model_target(states_next).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_rate * q_targets_next * (1 - dones)  # bellman
        q = self.model(states).gather(1, actions)

        loss = F.mse_loss(q, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.model, self.model_target, self.tau)

    def save(self, name):
        torch.save(self.model.state_dict(), f'{name}.pth')


def dqn_train(env, agent, n_episodes=2000, max_t=1000, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = 1.0
    eps_min = 0.01
    for i in tqdm(range(n_episodes)):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            state_next, reward, done, _ = env.step(action)
            agent.step(state, action, reward, state_next, done)
            state = state_next
            score += reward
            if done:
                break
        scores.append(score)
        scores_window.append(score)
        eps = max(eps_min, eps_decay*eps)
        if (i+1) % 100 == 0:
            print('\rEpisode {}\tAverage Score: {}'.format(i+1, np.mean(scores_window)))
        if np.mean(scores_window)>=200.0:
            torch.save(agent.model.state_dict(), 'checkpoint.pth')
            break
    return scores


if __name__ == "__main__":
    """
    states:
        inventory
        mid_price_returns
        best_ref_spread
        best_size
        rvol
    
    actions:
        ask spread
        bid spread
        hedging coefficient
    """

    NUM_SIM = 5
    SIM_DAYS = 10
    NUM_SEEDS = 5

    MU = 0.2
    SIGMA = 0.5
    MID_PRICE = 75

    NUM_SIMPLE = 20
    NUM_MOM = 5
    NUM_REVERSE = 5
    NUM_INFORMED = 2

    seeds = np.arange(NUM_SEEDS)

    ecn_book = pd.read_csv("MSFT.csv", index_col=0, parse_dates=True)
    epoch_time = pd.offsets.Minute(15)
    params_df = fit_ref_curve(ecn_book, epoch_time)
    ecn_book['bucket_label'] = ecn_book.index.floor(epoch_time).time
    ecn_book['spread'] = (ecn_book['ask_price1'] - ecn_book['bid_price1']).round(6)
    bucket_g = ecn_book.groupby('bucket_label')
    gm_coef_df = bucket_g.apply(fit_best_volume)
    ref_curve_df = sample_ref_size_curve(SIM_DAYS, params_df, gm_coef_df)

    agent = DQNAgent(n_state=5, n_action=8, discount_rate=0.99, tau=1e-3, learning_rate=5e-4, seed=0)
    env = DQNEnv(agent_params={'random': {'num': 1},
                               'persistent': {'num': 1},
                               'adaptive': {'num': 1},
                               'num_simple': NUM_SIMPLE,
                               'num_mom': NUM_MOM,
                               'num_reverse': NUM_REVERSE,
                               'numm_informed': NUM_INFORMED},
                 market_params={'init_mid_price': MID_PRICE, 'mu': MU, 'sig': SIGMA, 'price_impact_coeff': 0.3},
                 ref_curve=sample_ref_size_curve(SIM_DAYS, params_df, gm_coef_df, False),
                 ref_best_spread=sample_ref_spread(SIM_DAYS, ecn_book, False),
                 T=26 * SIM_DAYS, rl_agent=agent, seed=None, verbose=False)
    scores = dqn_train(env, agent)
