from abc import ABC, abstractmethod
import numpy as np


class InvestorAgent(metaclass=ABC):

    def __init__(self, name=None):
        self.name = name

    @staticmethod
    def trade_size(a=10, k=1, spread=1):
        return min(np.random.poisson(a * np.exp(-k * spread)), 100)

    @abstractmethod
    def submit_trade(self, *args, **kwargs):
        pass


class SimpleInvestorAgent(InvestorAgent):
    def __init__(self, name=None, buy_tendency=0.5):
        super().__init__(name=name)
        self.buy_tendency=buy_tendency

    def submit_trade(self):
        rand_num = np.random.uniform(0, 1)
        return 1 if rand_num > self.buy_tendency else -1


class MomentumInvestorAgent(InvestorAgent):
    def __init__(self, name, lookback=10, lower_threshold=0.1, upper_threshold=0.3):
        super().__init__(name=name)
        self._lookback = lookback
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def submit_trade(self, prices):
        prices = np.array(prices)
        if len(prices) > self._lookback:
            rand_num = np.random.uniform(0, 1)
            rtn = prices[1:] / prices[:-1] - 1
            rtn = rtn[-self._lookback:]
            last_rtn = rtn[-1]
            mean_rtn = np.mean(rtn)
            std_rtn = np.std(rtn)
            if mean_rtn + std_rtn > last_rtn >= mean_rtn - std_rtn:
                return 0
            elif mean_rtn + std_rtn <= last_rtn < mean_rtn + 2 * std_rtn:
                return rand_num > self.upper_threshold
            elif mean_rtn + 2 * std_rtn <= last_rtn < mean_rtn + 3 * std_rtn:
                return rand_num > self.lower_threshold
            elif last_rtn >= mean_rtn + 3 * std_rtn:
                return 1
            elif mean_rtn - 2 * std_rtn <= last_rtn < mean_rtn - std_rtn:
                return -(rand_num > self.upper_threshold)
            elif last_rtn >= mean_rtn - 2 * std_rtn and last_rtn >= mean_rtn - 3 * std_rtn:
                return -(rand_num > self.lower_threshold)
            elif last_rtn <= mean_rtn - 3 * std_rtn:
                return -1
        else:
            rand_num = np.random.uniform(0, 1)
            return 1 if rand_num > 0.5 else -1


class MeanReversionInvestorAgent(InvestorAgent):
    def __init__(self, name, lookback=10, lower_threshold=0.1, upper_threshold=0.3):
        super().__init__(name=name)
        self._lookback = lookback
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def submit_trade(self, prices):
        prices = np.array(prices)
        if len(prices) > self._lookback:
            rand_num = np.random.uniform(0, 1)
            rtn = prices[1:] / prices[:-1] - 1
            rtn = rtn[-self._lookback:]
            last_rtn = rtn[-1]
            mean_rtn = np.mean(rtn)
            std_rtn = np.std(rtn)
            if mean_rtn + std_rtn > last_rtn >= mean_rtn - std_rtn:
                return 0
            elif mean_rtn + std_rtn <= last_rtn < mean_rtn + 2 * std_rtn:
                return -(rand_num > self.upper_threshold)
            elif mean_rtn + 2 * std_rtn <= last_rtn < mean_rtn + 3 * std_rtn:
                return -(rand_num > self.lower_threshold)
            elif last_rtn >= mean_rtn + 3 * std_rtn:
                return -1
            elif mean_rtn - std_rtn >= last_rtn > mean_rtn - 2 * std_rtn:
                return rand_num > self.upper_threshold
            elif mean_rtn - 2 * std_rtn > last_rtn >= mean_rtn - 3 * std_rtn:
                return rand_num > self.lower_threshold
            elif last_rtn < mean_rtn - 3 * std_rtn:
                return 1
        else:
            rand_num = np.random.uniform(0, 1)
            return 1 if rand_num > 0.5 else -1


class InformedInvestorAgent(InvestorAgent):
    def __init__(self, name=None):
        super().__init__(name=name)

    def submit_trade(self, current_price, next_price):
        if next_price > current_price:
            return 1
        elif next_price < current_price:
            return -1
        else:
            return 0
