import numpy as np
import pandas as pd
import random

from buildingblocks.environment import QLearningEnv
from rl_agent import RLAgent
from utils.calibration import sample_ref_size_curve, fit_ref_curve, fit_best_volume, sample_ref_spread


class QLearningAgent(RLAgent):
    def __init__(self, Q, n_action, name: str = None):
        super().__init__(name=name, n_action=n_action)
        self.Q = Q

    def act(self, state, eps=0.):
        self.state = state
        if random.random() > eps:
            action = np.argmax(self.Q[self.state])
        else:
            action = random.choice(np.arange(self.n_action))
        self.action = action
        return action


def qlearning_train(env, agent, n_episodes, alpha=0.99, discount_factor=0.9, epsilon=0.1, t_max=500):

    for i in range(n_episodes):
        state = env.reset()
        for t in range(t_max):
            action = agent.act(state, eps=epsilon)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            else:
                agent.Q[state, action] += alpha * (reward + discount_factor *
                                                   np.amax(agent.Q[next_state]) - agent.Q[state, action])
            state = next_state

    return agent


if __name__ == "__main__":
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

    agent = QLearningAgent(Q=np.zeros((2**5, 8)), n_action=8)
    env = QLearningEnv(agent_params={'random': {'num': 1},
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
    agent = qlearning_train(env, agent, n_episodes=100, alpha=0.99, discount_factor=0.9, epsilon=0.1, t_max=500)
