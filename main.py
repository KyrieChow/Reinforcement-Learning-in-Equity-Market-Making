import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from buildingblocks.simulation import Simulation
from utils.calibration import sample_ref_size_curve, fit_ref_curve, fit_best_volume, sample_ref_spread
from utils.analysis import sim_analysis
from models.qlearning_agent import QLearningAgent
from models.dqn_agent import DQNAgent


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

    sim_list_ra = []  # cache simulation results

    for seed in seeds:
        print(f'========== Seed {seed} ==========')
        for i in tqdm(range(NUM_SIM)):
            # print(f'========== Simulation # {i}===========')

            # Q learning
            Q = np.load('model.npy')
            rl_agent = QLearningAgent(Q, n_action=8, name='RL Agent')

            # # DQN
            # dqn_model = torch.load('model.pth')
            # rl_agent = DQNAgent(n_state=5, n_action=8, discount_rate=0.99, tau=1e-3, learning_rate=5e-4, name='RL agent')
            # rl_agent.load_model(dqn_model)

            sim = Simulation(agent_params={'random': {'num': 1},
                                           'persistent': {'num': 1},
                                           'adaptive': {'num': 1},
                                           'num_simple': NUM_SIMPLE,
                                           'num_mom': NUM_MOM,
                                           'num_reverse': NUM_REVERSE,
                                           'numm_informed': NUM_INFORMED},
                             market_params={'init_mid_price': MID_PRICE, 'mu': MU, 'sig': SIGMA,
                                            'price_impact_coeff': 0.3},
                             ref_curve=sample_ref_size_curve(SIM_DAYS, params_df, gm_coef_df, False),
                             ref_best_spread=sample_ref_spread(SIM_DAYS, ecn_book, False),
                             T=26 * SIM_DAYS,
                             rl_agent=rl_agent,
                             verbose=False)
            sim.run()

            sim_list_ra.append(sim)

    mm_eps_dict_ra, mm_pnl_dict_ra = sim_analysis(sim_list_ra, NUM_SIM, NUM_SEEDS, SIM_DAYS)
