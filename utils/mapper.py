import pandas as pd
import numpy as np


def state_mapper(state_info, thresholds=None):
    key_order = ['inventory', 'mid_price_returns', 'best_ref_spread',
                 'best_size', 'rvol']
    if thresholds is None:
        thresholds = {
            'inventory': 20,
            'mid_price_returns': 0.02,
            'best_ref_spread': 0.005,
            'best_size': 5,
            'rvol': 0.03
        }
    df_map = pd.DataFrame(data=np.arange(2 ** len(key_order)),
                          index=pd.MultiIndex.from_product([
                              [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
                          ]),
                          columns=['state'])
    state_key = []
    for key in key_order:
        state_key.append(int(state_info[key] > thresholds[key]))
    return df_map.loc[tuple(state_key), 'state']


def action_mapper(action_key):
    key_order = ['bid_spread', 'ask_spread', 'hedge_ratio']

    df_map = pd.MultiIndex.from_product([
        [0.01, 0.03], [0.01, 0.03], [0.1, 0.5]
    ]).to_frame()
    df_map.columns = key_order
    df_map = df_map.reset_index(drop=True)
    return df_map.loc[action_key].to_dict()
