from mm_agent import MarketMakerAgent
from utils.mapper import action_mapper


class RLAgent(MarketMakerAgent):
    def __init__(self, n_action, name: str = None):
        super().__init__(name=name)
        self.n_action = n_action
        self.action = 0

    def get_spread_bid(self, action=None):
        return action_mapper(action)['bid_spread']

    def get_spread_ask(self, action=None):
        return action_mapper(action)['ask_spread']

    def get_hedge(self, action=None):
        return action_mapper(action)['hedge_ratio']
