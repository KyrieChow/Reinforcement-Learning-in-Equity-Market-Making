from abc import ABC, abstractmethod
import numpy as np


class MarketMakerAgent(metaclass=ABC):
    def __init__(self, name: str = None):
        self.name = name
        self.mid_price_history = []
        self.spread_bid = []
        self.spread_ask = []
        self.eps_bid_history = []
        self.eps_ask_history = []
        self.hedge_coeffs = []
        self.hedge_volume = []
        self.hedge_cost = []
        self.inventory_history = []
        self.pnl_inventory = []  # not cumulative
        self.inventory_value = 0
        self.pnl_spread = []  # not cumulative
        self.normalized_pnl_spread = []
        self.market_volume_history = []
        self.net_flows = []
        self.buy_flows = []
        self.sell_flows = []

    def win_trade(self, is_best_buy: bool, is_best_sell: bool, buy_size: float, sell_size: float, mid_price: float,
                  best_buy_dollar_spread: float, best_sell_dollar_spread: float, reference_best_dollar_spread: float,
                  agent_buy_dollar_spread: float, agent_sell_dollar_spread: float, epoch_id: int):
        flow = buy_size * is_best_buy - sell_size * is_best_sell
        if len(self.mid_price_history) == 0:
            self.pnl_inventory.append(0)
        else:
            inventory_pnl = self.inventory_value * (mid_price - self.mid_price_history[-1])
            if len(self.pnl_inventory) <= epoch_id:
                self.pnl_inventory.append(inventory_pnl)
            else:
                self.pnl_inventory[-1] += inventory_pnl
        self.inventory_value += flow

        spread_pnl = buy_size * is_best_buy * best_buy_dollar_spread + \
                     sell_size * is_best_sell * best_sell_dollar_spread

        if len(self.pnl_spread) <= epoch_id:
            # new epoch
            self.inventory_history.append(self.inventory_value)
            self.pnl_spread.append(spread_pnl)
            self.normalized_pnl_spread.append(spread_pnl / reference_best_dollar_spread)
            self.mid_price_history.append(mid_price)
            self.spread_bid.append(agent_buy_dollar_spread / reference_best_dollar_spread - 1)
            self.spread_ask.append(agent_sell_dollar_spread / reference_best_dollar_spread - 1)
            self.net_flows.append(buy_size * is_best_buy - sell_size * is_best_sell)
            self.buy_flows.append(buy_size * is_best_buy)
            self.sell_flows.append(sell_size * is_best_sell)
            self.market_volume_history.append(buy_size + sell_size)
        else:
            # current epoch
            self.inventory_history[-1] = self.inventory_value
            self.pnl_spread[-1] += spread_pnl
            self.normalized_pnl_spread[-1] = spread_pnl / reference_best_dollar_spread
            self.mid_price_history[-1] = mid_price
            self.spread_bid[-1] = agent_buy_dollar_spread / reference_best_dollar_spread - 1
            self.spread_ask[-1] = agent_sell_dollar_spread / reference_best_dollar_spread - 1
            self.net_flows[-1] += buy_size * is_best_buy - sell_size * is_best_sell
            self.buy_flows[-1] += buy_size * is_best_buy
            self.sell_flows[-1] += sell_size * is_best_sell
            self.market_volume_history[-1] += buy_size + sell_size

    def submit_hedge(self, hedge_ratio: float, buy_curve: list, sell_curve: list, size_curve: list, mid_price: float):
        cur_inventory_dir = int(self.inventory_history[-1] > 0)
        hedge_size = np.absolute(hedge_ratio * cur_inventory_dir)
        to_hedge_size = hedge_size
        hedge_cost = 0
        lob_level = 0
        max_lob_level = len(size_curve)

        while to_hedge_size > 0 and lob_level < max_lob_level:
            hedged_size = min(size_curve[lob_level], to_hedge_size)
            to_hedge_size -= hedged_size
            hedge_cost += (
                hedged_size * (buy_curve[lob_level] - mid_price) if cur_inventory_dir < 0 else hedged_size * (
                        sell_curve[lob_level] - mid_price))
            lob_level += 1

        # if all lob level is used up, trade at last level
        if to_hedge_size > 0:
            hedge_cost += (to_hedge_size * (buy_curve[-1] - mid_price) if cur_inventory_dir < 0 else
                           to_hedge_size * (sell_curve[-1] - mid_price))

        self.inventory_history[-1] -= hedge_size * cur_inventory_dir
        self.hedge_cost.append(hedge_cost)
        self.hedge_coeffs.append(hedge_ratio)

    @abstractmethod
    def get_spread_bid(self, action=None):
        pass

    @abstractmethod
    def get_spread_ask(self, action=None):
        pass

    @abstractmethod
    def get_hedge(self, action=None):
        pass
