from abc import ABC
from gym import Env
import pandas as pd
import numpy as np
import time

from models.simple_mm_agents import RandomAgent, PersistentAgent
from models.adaptive_mm_agent import AdaptiveAgent
from models.investor_agents import SimpleInvestorAgent, MomentumInvestorAgent, \
    MeanReversionInvestorAgent, InformedInvestorAgent
from buildingblocks.epoch import Epoch
from utils.mapper import state_mapper


def get_reward(spread_pnl, inventory_pnl, rvol, cur_inventory):
    risk_aversion = -0.5 * np.abs(cur_inventory) * rvol
    reward = spread_pnl + risk_aversion + inventory_pnl
    return reward


class Environment(Env, ABC):
    def __init__(self, agent_params, market_params, ref_curve: pd.DataFrame, ref_best_spread=None,
                 T=26, rl_agent=None, seed=None, verbose=False, max_iter=1000):
        if seed is None:
            seed = int(time.time())
        np.random.seed(seed=seed)

        assert 'init_mid_price' in market_params, "Initial Mid Price Not Provided"
        assert 'mu' in market_params, "Drift Not Provided"
        assert 'sig' in market_params, "Volatility Not Provided"

        self.agent_params = agent_params
        self.market_params = market_params
        self.ref_curve = ref_curve
        self.ref_best_spread = list(ref_best_spread) if ref_best_spread is not None else None
        self.verbose = verbose
        self.dt = 1
        self.T = T
        self.epoch = T // self.dt

        self.num_simple = self.agent_params.get('num_simple', 20)
        self.num_mom = self.agent_params.get('num_mom', 0)
        self.num_reverse = self.agent_params.get('num_reverse', 0)
        self.num_informed = self.agent_params.get('num_informed', 0)

        self.num_investors = self.num_simple + self.num_mom + self.num_reverse + self.num_informed

        self.rl_agent = rl_agent
        self.max_iter = max_iter
        self.count = 0

        self.reset_env()

    def get_next_gbm_midprice(self):
        return self.cur_mid_price * np.exp(self.mu * self.dt - 0.5 * self.sig**2 *self.dt + \
                                           self.sig * self.dt ** 0.5 * np.random.normal())

    def get_next_epoch(self, action=None):
        lastEpoch = self.epochs[-1] if len(self.epochs) > 0 else None
        ref_best_spread = None if self.ref_best_spread is None else self.ref_best_spread[len(self.epochs)]

        new_epoch = Epoch(len(self.epochs), self.mid_prices, self.mm_agent_list, self.investors,
                          self.ref_curve.loc[len(self.epochs), :].to_list(),
                          ref_best_spread, self.rvol[-1], lastEpoch, self.verbose, self.next_gbm_price,
                          action=action)
        self.epochs.append(new_epoch)
        self.epochs[-1].run()

    def get_snr(self):
        """
        Get signal-to-noise ratio of the simulation
        :return:
        """
        sum_abs_chng = 0
        for i in range(1, len(self.mid_prices)):
            sum_abs_chng += abs(self.mid_prices[i] - self.mid_prices[i - 1])

        return abs(self.mid_prices[-1] - self.mid_prices[0]) / sum_abs_chng

    def reset_env(self):
        self.cur_mid_price = self.market_params['init_mid_price']
        self.price_impact_coeff = self.market_params.get('price_impact_coeff', 0)
        self.mu, self.sig = self.market_params['mu'] / 252 / 26, self.market_params['sig'] / 252 / 26
        self.rvol_window_len = 10
        self.rvol = [np.nan]
        self.next_gbm_price = self.get_next_gbm_midprice()

        self.mm_agent_list = []
        self.investor_agent_list = []
        self.epochs = []

        self.investors = []
        self.mid_prices = [self.cur_mid_price]

        # add random agents
        if 'random' not in self.agent_params:
            self.agent_params['random'] = {}
            num_ra = 0
        else:
            num_ra = self.agent_params['random'].get('num', 0)
            eps_min = self.agent_params['random'].get('eps_min', None)
            eps_max = self.agent_params['random'].get('eps_max', None)
            for i in range(num_ra):
                self.mm_agent_list.append(RandomAgent(name='Random_{}'.format(i), eps_min=eps_min, eps_max=eps_max))

        # add persistent agents
        if 'persistent' not in self.agent_params:
            self.agent_params['persistent'] = {}
            num_pa = 0
        else:
            num_pa = self.agent_params['persistent'].get('num', 0)
            if 'params' not in self.agent_params['persistent']:
                self.agent_params['persistent']['params'] = []
            for i, pa_dict in enumerate(self.agent_params['persistent']['params']):
                self.mm_agent_list.append(PersistentAgent(name='Persistent_{}'.format(i), eps_bid=pa_dict['sb'],
                                                          eps_ask=pa_dict['sa'], hedge_coeff=pa_dict['hc']))
            for i in range(num_pa - len(self.agent_params['persistent']['params'])):
                # default params
                self.mm_agent_list.append(PersistentAgent(name='Persistent_{}'.format(
                    i + len(self.agent_params['persistent']['params']))))

        # add adaptive agents
        if 'adaptive' not in self.agent_params:
            self.agent_params['adaptive'] = {}
            num_aa = 0
        else:
            num_aa = self.agent_params['adaptive'].get('num', 0)
            if 'params' not in self.agent_params['persistent']:
                self.agent_params['adaptive']['params'] = []

            for i, aa_dict in enumerate(self.agent_params['adaptive']['params']):
                self.mm_agent_list.append(AdaptiveAgent(name='Adaptive_{}'.format(i), tolerance=aa_dict.get('tol'),
                                                        sigma=aa_dict.get('sigma'), gamma=aa_dict.get('gamma'),
                                                        market_share_target=aa_dict.get('ms_target'),
                                                        kde_range=max(50, 50 * self.num_investors / 20), num_aa=num_aa))
            for i in range(num_aa - len(self.agent_params['adaptive']['params'])):
                # default params
                self.mm_agent_list.append(AdaptiveAgent(name='Adaptive_{}'.format(i + len(self.agent_params['adaptive']['params'])),
                                                        kde_range=max(50, 50 * self.num_investors / 20), num_aa=num_aa))

        # add rl_agent
        if self.rl_agent is not None:
            self.rl_agent.reset()
            self.mm_agent_list.append(self.rl_agent)

        # add investor agent
        for i in range(self.num_simple):
            self.investors.append(SimpleInvestorAgent(name=f'Simple Investor {i}', buy_tendency=0.5))
        for i in range(self.num_mom):
            self.investors.append(MomentumInvestorAgent(name=f'Momentum Investor {i}'))
        for i in range(self.num_reverse):
            self.investors.append(MeanReversionInvestorAgent(name=f'Mean-Reverse Investor {i}'))
        for i in range(self.num_informed):
            self.investors.append(InformedInvestorAgent(name=f'Informed Investor {i}'))

        if num_ra == 0 and num_pa == 0 and num_aa == 0 and self.rl_agent is None:
            raise Exception

    def render(self, action, reward):
        print(f"Round : {self.count}\nAction: {action}\nReward: {reward}")
        print("=============================================================================")


class QLearningEnv(Environment):
    def __init__(self, agent_params, market_params, ref_curve: pd.DataFrame, ref_best_spread=None,
                 T=26, rl_agent=None, seed=None, verbose=False, thresholds=None, n_state=None):
        super().__init__(agent_params=agent_params, market_params=market_params,
                         ref_curve=ref_curve, ref_best_spread=ref_best_spread,
                         T=T, rl_agent=rl_agent, seed=seed, verbose=verbose)
        self.thresholds = thresholds
        self.n_state = n_state

    def reset(self):
        self.reset_env()
        state = 0
        return state

    def step(self, action):
        done = False

        self.get_next_epoch(action)
        self.cur_mid_price = self.price_impact_coeff * self.epochs[-1].get_mid_price() + (
                1 - self.price_impact_coeff) * self.next_gbm_price
        self.next_gbm_price = self.get_next_gbm_midprice()
        self.mid_prices.append(self.cur_mid_price)

        mid_prices_temp = np.array(self.mid_prices[-self.rvol_window_len - 1:])
        mid_prices_rtn = mid_prices_temp[1:] - mid_prices_temp[:-1]
        self.rvol.append(sum(mid_prices_rtn ** 2) * self.rvol_window_len / len(mid_prices_rtn))

        next_state_info = {
            'inventory': self.rl_agent.inventory_value,
            'mid_price_returns': mid_prices_rtn,
            'best_ref_spread': self.epochs[-1].ref_best_spread,
            'best_size': self.epochs[-1].size_reference_curve[0],
            'rvol': self.rvol[-1],
        }
        next_state = state_mapper(next_state_info, self.thresholds)
        info = {}

        reward = get_reward(spread_pnl=self.rl_agent.pnl_spread[-1],
                            inventory_pnl=self.rl_agent.inventory_pnl[-1],
                            rvol=self.rvol[-1], cur_inventory=self.rl_agent.inventory_value)

        self.count += 1
        if self.count >= self.max_iter:
            done = True

        self.render(action, reward)

        return next_state, reward, done, info


class DQNEnv(Environment):
    def __init__(self, agent_params, market_params, ref_curve: pd.DataFrame, ref_best_spread=None,
                 T=26, rl_agent=None, seed=None, verbose=False):
        super().__init__(agent_params=agent_params, market_params=market_params,
                         ref_curve=ref_curve, ref_best_spread=ref_best_spread,
                         T=T, rl_agent=rl_agent, seed=seed, verbose=verbose)

    def reset(self):
        self.reset_env()
        state_info = {
            'inventory': 0,
            'mid_price_returns': 0,
            'best_ref_spread': 0,
            'best_size': 0,
            'rvol': 0,
        }
        state = np.array(list(state_info.values()))
        return state

    def step(self, action):
        done = False

        self.get_next_epoch(action)
        self.cur_mid_price = self.price_impact_coeff * self.epochs[-1].get_mid_price() + (
                1 - self.price_impact_coeff) * self.next_gbm_price
        self.next_gbm_price = self.get_next_gbm_midprice()
        self.mid_prices.append(self.cur_mid_price)

        mid_prices_temp = np.array(self.mid_prices[-self.rvol_window_len - 1:])
        mid_prices_rtn = mid_prices_temp[1:] - mid_prices_temp[:-1]
        self.rvol.append(sum(mid_prices_rtn ** 2) * self.rvol_window_len / len(mid_prices_rtn))

        next_state_info = {
            'inventory': self.rl_agent.inventory_value,
            'mid_price_returns': mid_prices_rtn,
            'best_ref_spread': self.epochs[-1].ref_best_spread,
            'best_size': self.epochs[-1].size_reference_curve[0],
            'rvol': self.rvol[-1],
        }
        next_state = np.array(list(next_state_info.values()))

        info = {}

        reward = get_reward(spread_pnl=self.rl_agent.pnl_spread[-1],
                            inventory_pnl=self.rl_agent.inventory_pnl[-1],
                            rvol=self.rvol[-1], cur_inventory=self.rl_agent.inventory_value)

        self.count += 1
        if self.count >= self.max_iter:
            done = True

        self.render(action, reward)

        return next_state, reward, done, info
