import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.preprocessing import PolynomialFeatures


def clean_order_book(order_book, time_slice=('09:40', '15:50'), freq='100ms'):
    """clean the order book data pulled from one ticker:
        -remove data outside the time slice
        -resample to given fixed frequency
    note: resample will resulted in additional data on holiday, it won't change results, but need to removed for efficiency

    Parameters
    ----------
    order_book : TYPE
        oneTick order book data.
    time_slice : TYPE, optional
        only include data bewteen the time slice. The default is ('09:40','15:50').
    freq : TYPE, optional
        frequency of the output order book data. The default is '100ms'.

    Returns
    -------
    order_book_df : TYPE
        fixed frequency order book data.


    """

    order_book = order_book.between_time(time_slice[0], time_slice[1]).copy()

    order_book.columns = [col.lower() for col in order_book.columns]
    if 'symbol_name' in order_book.columns:
        order_book.drop(columns='symbol_name', inplace=True)
    if 'mid_quote' not in order_book.columns:
        order_book['mid_quote'] = (order_book.ask_price1 + order_book.bid_price1) / 2
    order_book = order_book.astype(float)

    order_bin = order_book.resample(freq)

    order_book_df = order_bin.last()

    order_book_df = order_book_df.between_time(time_slice[0], time_slice[1]).copy()
    order_book_df.fillna(method='ffill', inplace=True)

    return order_book_df


def scale_2_stack(scale_ask, ask_price_list, ask_size_list):
    bps_ask = scale_ask[ask_price_list].stack().reset_index(level=1)
    bps_ask.columns = ['price_level', 'price']
    vol_ask = scale_ask[ask_size_list].stack().reset_index(level=1)
    vol_ask.columns = ['size_level', 'size']
    stack_ask = bps_ask[['price']].copy()
    stack_ask['size'] = vol_ask['size'].values
    return stack_ask


def fit_bucket(bucket, ndegree=2):
    poly = PolynomialFeatures(ndegree)

    X = poly.fit_transform(bucket[['price']])[:, 1:]
    Y = bucket['size'] - 1

    ols_res = sm.OLS(Y, X).fit()

    return ols_res.params


def fit_daily_bucket(daily_df, epoch_time=None,
                     round_lot=100, order_depth=10, tick=0.01,
                     bid_size_str='bid_size', ask_size_str='ask_size'):
    if epoch_time is None:
        epoch_time = pd.offsets.Minute(15)

    mid_price = daily_df['mid_quote']
    bb_price = daily_df['bid_price1']
    ba_price = daily_df['ask_price1']
    bb_size = daily_df['bid_size1']
    ba_size = daily_df['ask_size1']
    bid_size_list = [f'{bid_size_str}{i}' for i in range(1, order_depth + 1)]
    bid_price_list = [f'bid_price{i}' for i in range(1, order_depth + 1)]

    ask_size_list = [f'{ask_size_str}{i}' for i in range(1, order_depth + 1)]
    ask_price_list = [f'ask_price{i}' for i in range(1, order_depth + 1)]

    scale_ask = daily_df.loc[ba_size >= round_lot, ask_price_list + ask_size_list].copy()
    scale_bid = daily_df.loc[bb_size >= round_lot, bid_price_list + bid_size_list].copy()

    scale_ask[ask_price_list] = (scale_ask[ask_price_list].subtract(ba_price, axis=0)) / tick
    scale_ask[ask_size_list] = scale_ask[ask_size_list].div(ba_size, axis=0)

    scale_bid[bid_price_list] = -(scale_bid[bid_price_list].subtract(bb_price, axis=0)) / tick
    scale_bid[bid_size_list] = scale_bid[bid_size_list].div(bb_size, axis=0)

    stack_ask = scale_2_stack(scale_ask, ask_price_list, ask_size_list)
    stack_bid = scale_2_stack(scale_bid, bid_price_list, bid_size_list)
    stack_data = pd.concat([stack_ask, stack_bid]).sort_index()

    time_buckets = stack_data.resample(epoch_time)
    params_df = time_buckets.apply(fit_bucket)
    return params_df


def fit_ref_curve(ecn_book, epoch_time):
    """return size shape params by fitting 2 degree polynomial


    Parameters
    ----------
    ecn_book : TYPE
        DESCRIPTION.
    epoch_time : TYPE
        DESCRIPTION.

    Returns
    -------
    params_df : TYPE
        shape by time bucket
    """
    day_group = ecn_book.groupby(ecn_book.index.date, group_keys=False, as_index=False)
    params_df = day_group.apply(lambda df: fit_daily_bucket(df, epoch_time))
    params_df['date'] = params_df.index.date
    params_df['time'] = params_df.index.time
    params_df = params_df.reset_index(drop=True).set_index(['time']).sort_index()
    return params_df


def get_ref_curve(params, x=None):
    x = pd.Series(np.arange(0, 10)) if x is None else x
    b = params['x1']
    c = params['x2']
    y = x.apply(lambda n: max(1 + n * b + n ** 2 * c, 0))
    return y


def fit_best_volume(tmp_bucket, winsorize=True):
    bid_size = tmp_bucket['bid_size1']
    ask_size = tmp_bucket['ask_size1']
    size_data = pd.concat([ask_size, bid_size]).values
    if winsorize:
        q1 = np.quantile(size_data, 0.025)
        q2 = np.quantile(size_data, 0.975)
        ck = (size_data > q1) & (size_data < q2)
        size_data = size_data[ck]
    gamma_coef = pd.Series(stats.gamma.fit(size_data, floc=0))
    gamma_coef.index = ['a', 'loc', 'beta']
    return gamma_coef


def sample_ref_size_curve(day_num, params_df, gm_coef_df, verbose=True):
    """
    Sampling reference size curve
    :param day_num:
    :param params_df:
    :param gm_coef_df:
    :param verbose:
    :return:
    """
    num_bucket = len(gm_coef_df)

    shape_g = params_df.groupby('time', group_keys=False)
    res_list = []
    best_vol = []
    for i in range(day_num):
        one_day_sample = shape_g.apply(lambda df: df.sample(1, replace=True)).sort_index()
        res_list.append(one_day_sample)
        best_vol.append(
            gm_coef_df.apply(lambda r: stats.gamma.rvs(r['a'], r['loc'], scale=r['beta'], size=1)[0], axis=1))

    sample_params = pd.concat(res_list)
    sample_shapes = sample_params.apply(get_ref_curve, axis=1)
    best_ref_vol = pd.concat(best_vol)
    ref_curve_df = sample_shapes.copy().multiply(best_ref_vol, axis=0)
    ref_curve_df = ref_curve_df.reset_index().drop(columns=['time'])
    if verbose:
        print(f"Total number of day sample {day_num}; Total data length: {len(ref_curve_df)}")
    return ref_curve_df


def sample_ref_spread(day_num, ecn_book, verbose=True):
    """
    Sampling reference spread curve
    :param day_num:
    :param ecn_book:
    :param verbose:
    :return:
    """
    bucket_g = ecn_book.groupby('bucket_label')
    res_list = []
    for i in range(day_num):
        one_day_sample = bucket_g['spread'].apply(lambda ts: ts.sample(1)[0]).sort_index()
        res_list.append(one_day_sample)

    sample_spread = pd.concat(res_list).reset_index(drop=True)
    if verbose:
        print(f"Total number of day sample {day_num}; Total data length: {len(sample_spread)}")
    return sample_spread


if __name__ == "__main__":
    ecn_book = pd.read_csv("MSFT.csv", index_col=0, parse_dates=True)
    ecn_book2 = pd.read_csv("YELP.csv", index_col=0, parse_dates=True)
    ecn_book3 = pd.read_csv("GME.csv", index_col=0, parse_dates=True)
    # ecn_book.head()

    epoch_time = pd.offsets.Minute(15)
    params_df = fit_ref_curve(ecn_book, epoch_time)
    params_df2 = fit_ref_curve(ecn_book2, epoch_time)
    params_df3 = fit_ref_curve(ecn_book3, epoch_time)
    params_df.head()

    ecn_book['bucket_label'] = ecn_book.index.floor(epoch_time).time
    ecn_book['spread'] = (ecn_book['ask_price1'] - ecn_book['bid_price1']).round(6)
    ecn_book2['bucket_label'] = ecn_book2.index.floor(epoch_time).time
    ecn_book2['spread'] = (ecn_book2['ask_price1'] - ecn_book2['bid_price1']).round(6)
    ecn_book3['bucket_label'] = ecn_book3.index.floor(epoch_time).time
    ecn_book3['spread'] = (ecn_book3['ask_price1'] - ecn_book3['bid_price1']).round(6)
    bucket_g = ecn_book.groupby('bucket_label')
    bucket_g2 = ecn_book2.groupby('bucket_label')
    bucket_g3 = ecn_book3.groupby('bucket_label')
    gm_coef_df = bucket_g.apply(fit_best_volume)
    gm_coef_df2 = bucket_g2.apply(fit_best_volume)
    gm_coef_df3 = bucket_g3.apply(fit_best_volume)
    gm_coef_df.head()

    day_num = 10
    # MSFT
    ref_curve_df = sample_ref_size_curve(day_num, params_df, gm_coef_df)
    ref_spread = sample_ref_spread(day_num, ecn_book)

    # YELP
    ref_curve_df2 = sample_ref_size_curve(day_num, params_df2, gm_coef_df2)
    ref_spread2 = sample_ref_spread(day_num, ecn_book2)
    # GME
    ref_curve_df3 = sample_ref_size_curve(day_num, params_df3, gm_coef_df3)
    ref_spread3 = sample_ref_spread(day_num, ecn_book3)
    ref_spread.head()
