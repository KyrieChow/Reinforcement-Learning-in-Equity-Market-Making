import numpy as np

from mm_agent import MarketMakerAgent


class RandomAgent(MarketMakerAgent):
    def __init__(self, name: str = None, eps_min: float = None, eps_max: float = None):
        super().__init__(name=name)
        self.eps_min = eps_min or 0
        self.eps_max = eps_max or 0.5

    def get_spread_bid(self, action=None):
        eps_bid = np.random.uniform(low=self.eps_min, high=self.eps_max)
        return eps_bid

    def get_spread_ask(self, action=None):
        eps_ask = np.random.uniform(low=self.eps_min, high=self.eps_max)
        return eps_ask

    def get_hedge(self, action=None):
        hedge_coeff = np.random.uniform(low=0, high=1)
        return hedge_coeff


class PersistentAgent(MarketMakerAgent):
    def __init__(self, name: str = None, eps_bid: float = None, eps_ask: float = None, hedge_coeff: float = None):
        super().__init__(name=name)

        self.eps_bid = eps_bid or 0.3
        self.eps_ask = eps_ask or 0.3
        self.hedge_coeff = hedge_coeff or 0.1

    def get_spread_bid(self, action=None):
        return self.eps_bid

    def get_spread_ask(self, action=None):
        return self.eps_ask

    def get_hedge(self, action=None):
        return self.hedge_coeff
