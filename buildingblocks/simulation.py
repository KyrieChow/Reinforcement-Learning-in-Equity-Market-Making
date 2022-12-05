import numpy as np
import pandas as pd

from environment import Environment
from models.qlearning_agent import QLearningAgent
from models.dqn_agent import DQNAgent
from utils.mapper import state_mapper


class Simulation(Environment):
    def __init__(self, agent_params, market_params, ref_curve: pd.DataFrame, ref_best_spread=None,
                 T=26, rl_agent=None, seed=None, verbose=False, thresholds=None):
        super().__init__(agent_params=agent_params, market_params=market_params,
                         ref_curve=ref_curve, ref_best_spread=ref_best_spread,
                         T=T, rl_agent=rl_agent, seed=seed, verbose=verbose)
        self.thresholds = thresholds

    def run(self):
        state_info = {
            'inventory': 0,
            'mid_price_returns': 0,
            'best_ref_spread': 0,
            'best_size': 0,
            'rvol': 0,
        }
        for i in range(self.epoch):
            if self.verbose:
                print('\nEpoch {}'.format(i + 1))

            if self.rl_agent is None:
                action = None
            else:
                if isinstance(self.rl_agent, QLearningAgent):
                    state = state_mapper(state_info, self.thresholds)
                    action = self.rl_agent.act(state)
                elif isinstance(self.rl_agent, DQNAgent):
                    state = np.array(list(state_info.values()))
                    action = self.rl_agent.act(state)
                else:
                    raise NotImplementedError

            self.get_next_epoch(action=action)
            self.cur_mid_price = self.price_impact_coeff * self.epochs[-1].get_mid_price() + (
                    1 - self.price_impact_coeff) * self.next_gbm_price
            self.next_gbm_price = self.get_next_gbm_midprice()
            self.mid_prices.append(self.cur_mid_price)
            mid_prices_temp = np.array(self.mid_prices[-self.rvol_window_len - 1:])
            mid_prices_rtn = mid_prices_temp[1:] - mid_prices_temp[:-1]

            if len(self.mid_prices) > self.rvol_window_len:
                self.rvol.append(sum(mid_prices_rtn ** 2))
            else:
                self.rvol.append(np.nan)
            state_info = {
                'inventory': self.rl_agent.inventory_value,
                'mid_price_returns': mid_prices_rtn,
                'best_ref_spread': self.epochs[-1].ref_best_spread,
                'best_size': self.epochs[-1].size_reference_curve[0],
                'rvol': self.rvol[-1],
            }
