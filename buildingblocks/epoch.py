import numpy as np

from models.adaptive_mm_agent import AdaptiveAgent
from models.investor_agents import SimpleInvestorAgent, MomentumInvestorAgent, MeanReversionInvestorAgent, InformedInvestorAgent

MAX_LOB_LEVEL = 10


class Epoch:
    def __init__(self, epoch_id, mid_prices, agent_list, investor_list, ref_curve: list, ref_best_spread: float = None,
                 rvol=np.nan, last_epoch=None, verbose=False, next_mid_price=None, action=None):
        self.epoch_id = epoch_id
        self.mid_price = mid_prices[-1]
        self.mid_prices = mid_prices
        self.next_mid_price = next_mid_price
        self.rvol = rvol
        self.agent_list = agent_list
        self.last_epoch = last_epoch
        self.investor_list = investor_list
        self.ref_best_spread = ref_best_spread or max(1e-6, np.random.normal(loc=0.01, scale=0.005))
        self.action = action

        self.verbose = verbose

        self.size_reference_curve = ref_curve
        self.set_price_reference_curve()

        self.investor_buys = 0
        self.investor_sells = 0

        self.agent_buy_quotes = {}
        self.agent_sell_quotes = {}
        self.agent_price_level = {}
        self.agent_spread_eps = {}
        for agent in self.agent_list:
            if isinstance(agent, AdaptiveAgent):
                agent.set_ref_curve(spread_reference_curve=self.spread_curve,
                                    size_reference_curve=self.size_reference_curve)
            self.agent_price_level[agent] = 0
            eps_bid = agent.get_spread_bid(self.action)
            eps_ask = agent.get_spread_ask(self.action)
            self.agent_spread_eps[agent] = {'bid': eps_bid,
                                            'ask': eps_ask}
            agent.eps_bid_history.append(eps_bid)
            agent.eps_ask_history.append(eps_ask)

    def set_price_reference_curve(self):
        self.spread_curve = [self.ref_best_spread / 2 + i * 0.01 for i in range(10)]
        self.buy_price_reference_curve = [self.mid_price - spread for spread in self.spread_curve]
        self.sell_price_reference_curve = [self.mid_price + spread for spread in self.spread_curve]

    def run(self):
        # simulate trades and movement of price
        for investor in self.investor_list:
            if isinstance(investor, SimpleInvestorAgent):
                direction = investor.submit_trade()
            elif isinstance(investor, MomentumInvestorAgent) or isinstance(investor, MeanReversionInvestorAgent):
                direction = investor.submit_trade(prices=self.mid_prices)
            elif isinstance(investor, InformedInvestorAgent):
                direction = investor.submit_trade(current_price=self.mid_price, next_price=self.next_mid_price)
            if direction == 1:
                self.investor_buys += investor.trade_size(a=10, k=1, spread=self.spread_curve[0])
            elif direction == -1:
                self.investor_sells += investor.trade_size(a=10, k=1, spread=self.spread_curve[0])
        reference_best_spread = self.spread_curve[0]

        # best_buy_dollar_spread, cur_best_buy_agent, best_buy_size = 1, None, None
        # best_sell_dollar_spread, cur_best_sell_agent, best_sell_size = 1, None, None
        if self.verbose:
            print('Mid Price: {}'.format(self.mid_price))
            print('Quotes by Agents:')

        buy_quantity_left = self.investor_buys
        sell_quantity_left = self.investor_sells

        final_buy_dollar_spread = 0
        final_sell_dollar_spread = 0
        final_buy_size = 0
        final_sell_size = 0
        while buy_quantity_left > 0 or sell_quantity_left > 0:
            if self.verbose:
                print(self.agent_price_level)
            best_buy_dollar_spread, cur_best_buy_agent, best_buy_size = 999, None, 0
            best_sell_dollar_spread, cur_best_sell_agent, best_sell_size = 999, None, 0
            for agent in self.agent_list:
                self.agent_buy_quotes[agent] = (1 + self.agent_spread_eps[agent]['bid']) * self.spread_curve[
                    self.agent_price_level[agent]]
                self.agent_sell_quotes[agent] = (1 + self.agent_spread_eps[agent]['ask']) * self.spread_curve[
                    self.agent_price_level[agent]]
                if self.verbose:
                    print("Agent {}: Bid Spread - {}, Ask Spread - {}".format(agent.name,
                                                                              self.agent_buy_quotes[agent],
                                                                              self.agent_sell_quotes[agent]))

                best_buy_dollar_spread, cur_best_buy_agent, best_buy_size = self.agent_trade(
                    agent=agent,
                    quantity_left=buy_quantity_left,
                    best_dollar_spread=best_buy_dollar_spread,
                    cur_best_agent=cur_best_buy_agent,
                    best_size=best_buy_size,
                    investor_action=self.investor_buys,
                    agent_quotes=self.agent_buy_quotes)

                best_sell_dollar_spread, cur_best_sell_agent, best_sell_size = self.agent_trade(
                    agent=agent,
                    quantity_left=sell_quantity_left,
                    best_dollar_spread=best_sell_dollar_spread,
                    cur_best_agent=cur_best_sell_agent,
                    best_size=best_sell_size,
                    investor_action=self.investor_sells,
                    agent_quotes=self.agent_sell_quotes)

            for agent in self.agent_list:
                is_best_buy = (agent == cur_best_buy_agent)
                is_best_sell = (agent == cur_best_sell_agent)
                # print('best_size:',best_buy_size,best_sell_size)
                agent.win_trade(is_best_buy=is_best_buy, is_best_sell=is_best_sell, buy_size=best_buy_size,
                                sell_size=best_sell_size, mid_price=self.mid_price,
                                best_buy_dollar_spread=best_buy_dollar_spread,
                                best_sell_dollar_spread=best_sell_dollar_spread,
                                reference_best_dollar_spread=reference_best_spread, agent_buy_dollar_spread=
                                self.agent_buy_quotes[agent], agent_sell_dollar_spread=self.agent_sell_quotes[agent],
                                epoch_id=self.epoch_id)

                if is_best_buy:
                    if self.verbose:
                        print('Agent: {} wins buy trade of size {} with price {}'.format(agent.name, best_buy_size,
                                                                                         self.mid_price - best_buy_dollar_spread))
                    final_buy_dollar_spread = best_buy_dollar_spread
                    final_buy_size += best_buy_size

                if is_best_sell:
                    if self.verbose:
                        print('Agent: {} wins sell trade of size {} with price {}'.format(agent.name, best_sell_size,
                                                                                          self.mid_price + best_sell_dollar_spread))
                    final_sell_dollar_spread = best_sell_dollar_spread
                    final_sell_size += best_sell_size

            buy_quantity_left -= best_buy_size
            sell_quantity_left -= best_sell_size
        self.mid_price = self.mid_price - final_buy_dollar_spread * final_buy_size / (
                    final_buy_size + final_sell_size) + \
                         final_sell_dollar_spread * final_sell_size / (final_buy_size + final_sell_size)

        for agent in self.agent_list:
            agent_hedge_ratio = agent.get_hedge()
            agent.submit_hedge(agent_hedge_ratio, self.buy_price_reference_curve, self.sell_price_reference_curve,
                               self.size_reference_curve, self.mid_price)

    def lob_shape(self, n):
        return -1 / 32 * n ** 2 + (1 / 4) * n + 1

    def get_mid_price(self):
        return self.mid_price

    def get_rvol(self):
        return self.rvol

    def agent_trade(self, agent, quantity_left, best_dollar_spread, cur_best_agent,
                    best_size, investor_action, agent_quotes):
        if quantity_left > 0 and agent_quotes[agent] < best_dollar_spread:
            if cur_best_agent is not None:
                self.agent_price_level[cur_best_agent] -= 1
            best_dollar_spread = agent_quotes[agent]
            cur_best_agent = agent
            best_size = min(self.size_reference_curve[self.agent_price_level[agent]], investor_action)
            self.agent_price_level[agent] += 1
            self.agent_price_level[agent] = min(self.agent_price_level[agent], MAX_LOB_LEVEL - 1)

        return best_dollar_spread, cur_best_agent, best_size
