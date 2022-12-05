import numpy as np
import random
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize, NonlinearConstraint

from mm_agent import MarketMakerAgent

KDE_ARRAY_SIZE = 1000
MAX_KDE_INPUT_SIZE = 52
OPTIMIZE_METHOD = 'BFGS'


class AdaptiveAgent(MarketMakerAgent):
    def __init__(self, name: str = None, tolerance: float = None, sigma: float = None, gamma: float = None,
                 market_share_target: float = None, kde_range: float = None, num_aa: int = None):

        super().__init__(name=name)
        self.num_aa = num_aa or 1
        self.initial_samples = int(4 * self.num_aa)
        self.tolerance = tolerance or 0.05

        # initialize
        self.net_flow_kde = None  # v_eps, design it as a 3d pdf with x-axis(epsilon_b,epsilon_s,v)
        self.buy_flows_kde = None
        self.sell_flows_kde = None
        self.spread_pnl_kde = None  # s_eps,, design it as a 3d pdf with x-axis(epsilon_b,epsilon_s,spread_pnl)
        self.spread_reference_curve = None  # S_ref
        self.size_reference_curve = None
        self.mkt_volume = None
        self.sigma = sigma or 0.1  # sigma of the mid price movement
        self.market_share_target = market_share_target or 0.5
        self.gamma = gamma or 2  # risk aversion parameter
        self.kde_range = kde_range or 100

    def set_ref_curve(self, spread_reference_curve, size_reference_curve):
        self.mkt_volume = self.market_volume_history[-1] if len(self.market_volume_history) > 0 else 0
        self.spread_reference_curve = spread_reference_curve  # S_ref
        self.size_reference_curve = size_reference_curve
        self.net_flow_kde = np.row_stack((self.spread_bid, self.spread_ask, self.net_flows))
        self.buy_flow_kde = np.row_stack((self.spread_bid, self.spread_ask, self.buy_flows))
        self.sell_flow_kde = np.row_stack((self.spread_bid, self.spread_ask, self.sell_flows))
        self.spread_pnl_kde = np.row_stack((self.spread_bid, self.spread_ask, self.normalized_pnl_spread))
        if self.net_flow_kde.shape[1] > self.initial_samples:
            self.net_flow_kernel = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(
                self.net_flow_kde[:, -MAX_KDE_INPUT_SIZE:].T)
            self.buy_flow_kernel = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(
                self.buy_flow_kde[:, -MAX_KDE_INPUT_SIZE:].T)
            self.sell_flow_kernel = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(
                self.sell_flow_kde[:, -MAX_KDE_INPUT_SIZE:].T)
            self.spread_pnl_kernel = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(
                self.spread_pnl_kde[:, -MAX_KDE_INPUT_SIZE:].T)

    def get_pdf(self, eps_b, eps_s, kde_type, z):
        if kde_type == 'net':
            kernal = self.net_flow_kernel
        elif kde_type == 'buy':
            kernal = self.buy_flow_kernel
        elif kde_type == 'sell':
            kernal = self.sell_flow_kernel
        elif kde_type == 'spread':
            kernal = self.spread_pnl_kernel
        else:
            raise NotImplementedError
        size_array = np.linspace(-self.kde_range, self.kde_range, KDE_ARRAY_SIZE) + z
        x = np.row_stack((np.ones((1, KDE_ARRAY_SIZE)) * eps_b, np.ones((1, KDE_ARRAY_SIZE)) * eps_s, size_array)).T
        size_pdf = kernal.score_samples(x)
        return size_pdf

    def get_mean(self, eps_b, eps_s, kde_type, z=0):
        try:
            size_array = np.linspace(-self.kde_range, self.kde_range, KDE_ARRAY_SIZE) + z
            size_pdf = self.get_pdf(eps_b, eps_s, kde_type, z)
            return np.sum(size_pdf * size_array) / (np.sum(size_pdf) + 0.001)
        except:
            # print(f'{self.name} kde error: singular')
            return 0

    def get_variance(self, eps_b, eps_s, kde_type, z=0):
        try:
            size_array = np.linspace(-self.kde_range, self.kde_range, KDE_ARRAY_SIZE) + z
            size_pdf = self.get_pdf(eps_b, eps_s, kde_type, z)
            return np.sum(size_pdf * (size_array ** 2)) / (np.sum(size_pdf) + 0.001) - \
                   np.sum(size_pdf * size_array / (np.sum(size_pdf) + 0.001)) ** 2
        except:
            # print(f'{self.name}kde error: singular')
            return 0

    def cost1(self, eps, mkt_share, mkt_volume):
        """
        cost function of general spread optimization
        :param eps:
        :param mkt_share:
        :param mkt_volume:
        :return:
        """
        if self.buy_flow_kde is not None and self.sell_flow_kde is not None and \
                self.buy_flow_kde.shape[1] > self.initial_samples and \
                self.sell_flow_kde.shape[1] > self.initial_samples:
            E_v = self.get_mean(eps, eps, kde_type='buy', z=0) + self.get_mean(eps, eps, kde_type='sell', z=0)
        else:
            E_v = 0
        return np.abs(mkt_share - E_v / (mkt_volume + 0.001))

    def cost2(self, eps, eps_star, gamma, sigma, z, skew_direction):
        """
        cost function of optimizing skewed spread
        :param eps:
        :param eps_star:
        :param gamma:
        :param sigma:
        :param z:
        :param skew_direction:
        :return:
        """
        if skew_direction == 'bid':
            eps1 = eps
            eps2 = eps_star
        else:  # ask
            eps1 = eps_star
            eps2 = eps

        if self.spread_pnl_kde is not None and self.spread_pnl_kde.shape[1] > self.initial_samples:
            E_s = self.get_mean(eps1, eps2, kde_type='spread', z=z)
            var_s = self.get_variance(eps1, eps2, kde_type='spread', z=z)
        else:
            E_s = 0
            var_s = 0

        if self.net_flow_kde is not None and self.net_flow_kde.shape[1] > self.initial_samples:
            E2_v_z = self.get_mean(eps1, eps2, kde_type='net', z=z) ** 2 + self.get_variance(
                eps1, eps2, kde_type='net', z=z)
        else:
            E2_v_z = 0

        return -self.spread_reference_curve[0] * E_s + gamma * (
                self.spread_reference_curve[0] ** 2 * var_s + sigma ** 2 * E2_v_z) ** 0.5

    def cost_hedge(self, x, z, eps_bid, eps_ask, sigma, gamma):
        """
        hedging cost
        Assume we first adapt the bid * ask spread, then do the hedge
        :param x:
        :param z:
        :param eps_bid:
        :param eps_ask:
        :param sigma:
        :param gamma:
        :return:
        """
        if self.net_flow_kde is not None and self.net_flow_kde.shape[1] > self.initial_samples:
            E2_v_z = self.get_mean(eps_bid, eps_ask, kde_type='net', z=z * (
                    1 - x)) ** 2 + self.get_variance(eps_bid, eps_ask, kde_type='net', z=z * (1 - x))
        else:
            E2_v_z = 0
        return np.abs(x * z) * self.spread_reference_curve[
            min(np.abs(int(x * z)), len(self.spread_reference_curve) - 1)] + \
               gamma * (sigma ** 2 * E2_v_z) ** 0.5

    def set_spread(self):
        """
        calculate generalized spread for bid/ask
        :return:
        """
        if self.net_flow_kde is None or self.net_flow_kde.shape[1] <= self.initial_samples:
            return random.uniform(0, 1)

        res0 = minimize(self.cost1, x0=np.array([0.01]), args=(
            self.market_share_target, self.mkt_volume), method=OPTIMIZE_METHOD)
        x = res0.x[0]

        def constraint(eps):
            return self.cost1(eps, self.market_share_target, self.mkt_volume) - \
                   self.cost1(x, self.market_share_target, self.mkt_volume)

        def minus_eps(eps):
            return -eps

        res1 = minimize(minus_eps, x0=np.array([0.01]),
                        constraints=NonlinearConstraint(constraint, lb=-self.tolerance, ub=self.tolerance),
                        method='SLSQP')  # constrained optimization
        return np.clip(res1.x[0], -1, 1)

    def get_spread_bid(self, action=None):
        """
        skewed bid spread
        :param action:
        :return:
        """
        self.eps_star = self.set_spread()  # Cautious, call one time when ask for spread
        # print("inventory: ", self.inventory_value)
        # print("eps_star",self.eps_star)
        if self.inventory_value <= 0:
            skew_direction = 'bid'
            z = self.inventory_value
            res = minimize(self.cost2, x0=np.array([0.01]),
                           args=(self.eps_star, self.gamma, self.sigma, z, skew_direction),
                           method=OPTIMIZE_METHOD)
            bid_spread = res.x[0]
            # print("bid_spread",bid_spread)
            return np.clip(bid_spread, -1, 1)
        else:
            return self.eps_star

    def get_spread_ask(self, action=None):
        """
        skewed ask spread
        :param action:
        :return:
        """
        if self.inventory_value > 0:
            skew_direction = 'ask'
            z = self.inventory_value
            res = minimize(self.cost2, x0=np.array([0.01]),
                           args=(self.eps_star, self.gamma, self.sigma, z, skew_direction),
                           method=OPTIMIZE_METHOD)
            ask_spread = res.x[0]
            # print("ask_spread",ask_spread)
            return np.clip(ask_spread, -1, 1)
        else:
            return self.eps_star

    def get_hedge(self, action=None):
        z = self.inventory_value
        res = minimize(self.cost_hedge, x0=np.array([0.01]),
                       args=(z, self.spread_bid[-1], self.spread_ask[-1], self.sigma, self.gamma),
                       method=OPTIMIZE_METHOD)
        hedge_ratio = res.x[0]
        return np.clip(hedge_ratio, 0, 1)
