#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#


"""
Utility functions
"""
import re
import time
import numpy as np
import urllib.request
import shutil
import os
import pandas as pd
from PIL import Image
from ta.momentum import *
from ta.trend import *
from ta.volume import *
from ta.others import *
from ta.volatility import *
from tqdm.auto import tqdm
from stockstats import StockDataFrame as sdf
from ta import *
from matplotlib import pyplot as plt
import winsound
import time


def seconds_to_minutes(seconds):
    return str(seconds // 60) + " minutes " + str(np.round(seconds % 60)) + " seconds"


def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text, seconds_to_minutes(seconds))


def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")


def download_save(url, path_to_save, logger=None):
    if logger:
        logger.append_log("Starting download " + re.sub(r'apikey=[A-Za-z0-9]+&', 'apikey=my_api_key&', url))
    else:
        print("Starting download " + re.sub(r'apikey=[A-Za-z0-9]+&', 'apikey=my_api_key&', url))
    urllib.request.urlretrieve(url, path_to_save)
    if logger:
        logger.append_log(path_to_save + " downloaded and saved")
    else:
        print(path_to_save + " downloaded and saved")


def remove_dir(path):
    shutil.rmtree(path)
    print(path, "deleted")
    # os.rmdir(path)


def save_array_as_images(x, img_width, img_height, path, file_names):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("deleted old files")

    os.makedirs(path)
    print("Image Directory created", path)
    x_temp = np.zeros((len(x), img_height, img_width))
    print("saving images...")
    stime = time.time()
    for i in tqdm(range(x.shape[0])):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))
        img = Image.fromarray(x_temp[i], 'RGB')
        img.save(os.path.join(path, str(file_names[i]) + '.png'))

    print_time("Images saved at " + path, stime)
    return x_temp


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp


def show_images(rows, columns, path):
    w = 15
    h = 15
    fig = plt.figure(figsize=(15, 15))
    files = os.listdir(path)
    for i in range(1, columns * rows + 1):
        index = np.random.randint(len(files))
        img = np.asarray(Image.open(os.path.join(path, files[index])))
        fig.add_subplot(rows, columns, i)
        plt.title(files[i], fontsize=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.imshow(img)
    plt.show()


def dict_to_str(d):
    return str(d).replace("{", '').replace("}", '').replace("'", "").replace(' ', '')


def cleanup_file_path(path):
    return path.replace('\\', '/').replace(" ", "_").replace(':', '_')


def white_noise_check(tags_list, logger=None, *pd_series_args):
    if len(tags_list) != len(pd_series_args):
        raise Exception("Length of tags_list and series params different. Should be same.")
    for idx, s in enumerate(pd_series_args):
        # logger.append_log("1st, 2nd element {}, {}".format(s.iloc[0], s.iloc[1]))
        m = s.mean()
        std = s.std()
        logger.append_log("mean & std for {} is {}, {}".format(tags_list[idx], m, std))


def plot(y, title, output_path, x=None):
    fig = plt.figure(figsize=(10, 10))
    # x = x if x is not None else np.arange(len(y))
    plt.title(title)
    if x is not None:
        plt.plot(x, y, 'o-')
    else:
        plt.plot(y, 'o-')
        plt.savefig(output_path)


def col1_gt_col2(col1, col2, df):
    compare_series = df[col1] > df[col2]
    print(df.iloc[compare_series[compare_series == True].index])


def sound_alert(repeat_count=5):
    duration = 1000  # millisecond
    freq = 440  # Hz
    for i in range(0, repeat_count):
        winsound.Beep(freq, duration)
        time.sleep(1)


def console_pretty_print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


############### Technical indicators ########################


# not used
def get_RSI(df, col_name, intervals):
    """
    stockstats lib seems to use 'close' column by default so col_name
    not used here.
    This calculates non-smoothed RSI
    """
    df_ss = sdf.retype(df)
    for i in intervals:
        df['rsi_' + str(i)] = df_ss['rsi_' + str(i)]

        del df['close_-1_s']
        del df['close_-1_d']
        del df['rs_' + str(i)]

        df['rsi_' + str(intervals[0])] = rsi(df['close'], i, fillna=True)
    print("RSI with stockstats done")


def get_RSI_smooth(df, col_name, intervals):
    """
    Momentum indicator
    As per https://www.investopedia.com/terms/r/rsi.asp
    RSI_1 = 100 - (100/ (1 + (avg gain% / avg loss%) ) )
    RSI_2 = 100 - (100/ (1 + (prev_avg_gain*13+avg gain% / prev_avg_loss*13 + avg loss%) ) )

    E.g. if period==6, first RSI starts from 7th index because difference of first row is NA
    http://cns.bu.edu/~gsc/CN710/fincast/Technical%20_indicators/Relative%20Strength%20Index%20(RSI).htm
    https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi
    Verified!
    """

    print("Calculating RSI")
    stime = time.time()
    prev_rsi = np.inf
    prev_avg_gain = np.inf
    prev_avg_loss = np.inf
    rolling_count = 0

    def calculate_RSI(series, period):
        # nonlocal rolling_count
        nonlocal prev_avg_gain
        nonlocal prev_avg_loss
        nonlocal rolling_count

        # num_gains = (series >= 0).sum()
        # num_losses = (series < 0).sum()
        # sum_gains = series[series >= 0].sum()
        # sum_losses = np.abs(series[series < 0].sum())
        curr_gains = series.where(series >= 0, 0)  # replace 0 where series not > 0
        curr_losses = np.abs(series.where(series < 0, 0))
        avg_gain = curr_gains.sum() / period  # * 100
        avg_loss = curr_losses.sum() / period  # * 100
        rsi = -1

        if rolling_count == 0:
            # first RSI calculation
            rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))
            # print(rolling_count,"rs1=",rs, rsi)
        else:
            # smoothed RSI
            # current gain and loss should be used, not avg_gain & avg_loss
            rsi = 100 - (100 / (1 + ((prev_avg_gain * (period - 1) + curr_gains.iloc[-1]) /
                                     (prev_avg_loss * (period - 1) + curr_losses.iloc[-1]))))
            # print(rolling_count,"rs2=",rs, rsi)

        # df['rsi_'+str(period)+'_own'][period + rolling_count] = rsi
        rolling_count = rolling_count + 1
        prev_avg_gain = avg_gain
        prev_avg_loss = avg_loss
        return rsi

    diff = df[col_name].diff()[1:]  # skip na
    for period in tqdm(intervals):
        df['rsi_' + str(period)] = np.nan
        # df['rsi_'+str(period)+'_own_1'] = np.nan
        rolling_count = 0
        res = diff.rolling(period).apply(calculate_RSI, args=(period,), raw=False)
        df['rsi_' + str(period)][1:] = res

    # df.drop(['diff'], axis = 1, inplace=True)
    print_time("Calculation of RSI Done", stime)


# not used: +1, ready to use
def get_IBR(df):
    return (df['close'] - df['low']) / (df['high'] - df['low'])


def get_williamR(df, col_name, intervals):
    """
    both libs gave same result
    Momentum indicator
    """
    stime = time.time()
    print("Calculating WilliamR")
    # df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        # df['wr_'+str(i)] = df_ss['wr_'+str(i)]
        df["wr_" + str(i)] = wr(df['high'], df['low'], df['close'], i, fillna=True)

    print_time("Calculation of WilliamR Done", stime)


def get_mfi(df, intervals):
    """
    momentum type indicator
    """

    stime = time.time()
    print("Calculating MFI")
    for i in tqdm(intervals):
        df['mfi_' + str(i)] = money_flow_index(df['high'], df['low'], df['close'], df['volume'], n=i, fillna=True)

    print_time("Calculation of MFI done", stime)


def get_SMA(df, col_name, intervals):
    """
    Momentum indicator
    """
    stime = time.time()
    print("Calculating SMA")
    df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        df[col_name + '_sma_' + str(i)] = df_ss[col_name + '_' + str(i) + '_sma']
        del df[col_name + '_' + str(i) + '_sma']

    print_time("Calculation of SMA Done", stime)


def get_EMA(df, col_name, intervals):
    """
    Needs validation
    Momentum indicator
    """
    stime = time.time()
    print("Calculating EMA")
    df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        df['ema_' + str(i)] = df_ss[col_name + '_' + str(i) + '_ema']
        del df[col_name + '_' + str(i) + '_ema']
        # df["ema_"+str(intervals[0])+'_1'] = ema_indicator(df['close'], i, fillna=True)

    print_time("Calculation of EMA Done", stime)


def get_WMA(df, col_name, intervals, hma_step=0):
    """
    Momentum indicator
    """
    stime = time.time()
    if (hma_step == 0):
        # don't show progress for internal WMA calculation for HMA
        print("Calculating WMA")

    def wavg(rolling_prices, period):
        weights = pd.Series(range(1, period + 1))
        return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

    temp_col_count_dict = {}
    for i in tqdm(intervals, disable=(hma_step != 0)):
        res = df[col_name].rolling(i).apply(wavg, args=(i,), raw=False)
        # print("interval {} has unique values {}".format(i, res.unique()))
        if hma_step == 0:
            df['wma_' + str(i)] = res
        elif hma_step == 1:
            if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
            else:
                temp_col_count_dict['hma_wma_' + str(i)] = 0
            # after halving the periods and rounding, there may be two intervals with same value e.g.
            # 2.6 & 2.8 both would lead to same value (3) after rounding. So save as diff columns
            df['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
        elif hma_step == 3:
            import re
            expr = r"^hma_[0-9]{1}"
            columns = list(df.columns)
            # print("searching", expr, "in", columns, "res=", list(filter(re.compile(expr).search, columns)))
            df['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res

    if hma_step == 0:
        print_time("Calculation of WMA Done", stime)


def get_HMA(df, col_name, intervals):
    import re
    stime = time.time()
    print("Calculating HMA")
    expr = r"^wma_.*"

    if len(list(filter(re.compile(expr).search, list(df.columns)))) > 0:
        print("WMA calculated already. Proceed with HMA")
    else:
        print("Need WMA first...")
        get_WMA(df, col_name, intervals)

    intervals_half = np.round([i / 2 for i in intervals]).astype(int)

    # step 1 = WMA for interval/2
    # this creates cols with prefix 'hma_wma_*'
    get_WMA(df, col_name, intervals_half, 1)
    # print("step 1 done", list(df.columns))

    # step 2 = step 1 - WMA
    columns = list(df.columns)
    expr = r"^hma_wma.*"
    hma_wma_cols = list(filter(re.compile(expr).search, columns))
    rest_cols = [x for x in columns if x not in hma_wma_cols]
    expr = r"^wma.*"
    wma_cols = list(filter(re.compile(expr).search, rest_cols))

    df[hma_wma_cols] = df[hma_wma_cols].sub(df[wma_cols].values,
                                            fill_value=0)  # .rename(index=str, columns={"close": "col1", "rsi_6": "col2"})
    # df[0:10].copy().reset_index(drop=True).merge(temp.reset_index(drop=True), left_index=True, right_index=True)

    # step 3 = WMA(step 2, interval = sqrt(n))
    intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
    for i, col in tqdm(enumerate(hma_wma_cols)):
        # print("step 3", col, intervals_sqrt[i])
        get_WMA(df, col, [intervals_sqrt[i]], 3)
    df.drop(columns=hma_wma_cols, inplace=True)
    print_time("Calculation of HMA Done", stime)


def get_TRIX(df, col_name, intervals):
    """
    TA lib actually calculates percent rate of change of a triple exponentially
    smoothed moving average not Triple EMA.
    Momentum indicator
    Need validation!
    """
    stime = time.time()
    print("Calculating TRIX")
    df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        # df['trix_'+str(i)] = df_ss['trix_'+str(i)+'_sma']
        df['trix_' + str(i)] = trix(df['close'], i, fillna=True)

    # df.drop(columns=['trix','trix_6_sma',])
    print_time("Calculation of TRIX Done", stime)


def get_DMI(df, col_name, intervals):
    """
    trend indicator
    TA gave same/wrong result
    """
    stime = time.time()
    print("Calculating DMI")
    df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        # df['dmi_'+str(i)] = adx(df['high'], df['low'], df['close'], n=i, fillna=True)
        df['dmi_' + str(i)] = df_ss['adx_' + str(i) + '_ema']

    drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema', 'pdm_14',
                    'close_-1_s', 'tr', 'tr_14_smma', 'atr_14']
    # drop_columns = ['high_delta', 'um', 'low_delta', 'dm', 'pdm', 'pdm_14_ema',
    #                 'pdm_14', 'close_-1_s', 'tr', 'atr_14', 'pdi_14', 'pdi',
    #                 'mdm', 'mdm_14_ema', 'mdm_14', 'mdi_14', 'mdi', 'dx_14',
    #                 'dx', 'adx', 'adxr']
    expr1 = r'dx_\d+_ema'
    expr2 = r'adx_\d+_ema'
    import re
    drop_columns.extend(list(filter(re.compile(expr1).search, list(df.columns)[9:])))
    drop_columns.extend(list(filter(re.compile(expr2).search, list(df.columns)[9:])))
    df.drop(columns=drop_columns, inplace=True)
    print_time("Calculation of DMI done", stime)


def get_CCI(df, col_name, intervals):
    stime = time.time()
    print("Calculating CCI")
    df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        # df['cci_'+str(i)] = df_ss['cci_'+str(i)]
        df['cci_' + str(i)] = cci(df['high'], df['low'], df['close'], i, fillna=True)

    print_time("Calculation of CCI Done", stime)


def get_BB_MAV(df, col_name, intervals):
    """
    volitility indicator
    """

    stime = time.time()
    print("Calculating Bollinger Band MAV")
    df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        df['bb_' + str(i)] = bollinger_mavg(df['close'], n=i, fillna=True)

    print_time("Calculation of Bollinger Band MAV done", stime)


def get_CMO(df, col_name, intervals):
    """
    Chande Momentum Oscillator
    As per https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo

    CMO = 100 * ((Sum(ups) - Sum(downs))/ ( (Sum(ups) + Sum(downs) ) )
    range = +100 to -100

    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated

    return: None (adds the result in a column)
    """

    print("Calculating CMO")
    stime = time.time()

    def calculate_CMO(series, period):
        # num_gains = (series >= 0).sum()
        # num_losses = (series < 0).sum()
        sum_gains = series[series >= 0].sum()
        sum_losses = np.abs(series[series < 0].sum())
        cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
        return np.round(cmo, 3)

    diff = df[col_name].diff()[1:]  # skip na
    for period in tqdm(intervals):
        df['cmo_' + str(period)] = np.nan
        res = diff.rolling(period).apply(calculate_CMO, args=(period,), raw=False)
        df['cmo_' + str(period)][1:] = res

    print_time("Calculation of CMO Done", stime)


# not used. on close(12,16): +3, ready to use
def get_MACD(df):
    """
    Not used
    Same for both
    calculated for same 12 and 26 periods on close only!! Not different periods.
    creates colums macd, macds, macdh
    """
    stime = time.time()
    print("Calculating MACD")
    df_ss = sdf.retype(df)
    df['macd'] = df_ss['macd']
    # df['macd_'+str(i)] = macd(df['close'], fillna=True)

    del df['macd_']
    del df['close_12_ema']
    del df['close_26_ema']
    print_time("Calculation of MACD done", stime)


# not implemented. period 12,26: +1, ready to use
def get_PPO(df, col_name, intervals):
    """
    As per https://www.investopedia.com/terms/p/ppo.asp

    uses EMA(12) and EMA(26) to calculate PPO value

    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated

    return: None (adds the result in a column)

    calculated for same 12 and 26 periods only!!
    """
    stime = time.time()
    print("Calculating PPO")
    df_ss = sdf.retype(df)
    df['ema_' + str(12)] = df_ss[col_name + '_' + str(12) + '_ema']
    del df['close_' + str(12) + '_ema']
    df['ema_' + str(26)] = df_ss[col_name + '_' + str(26) + '_ema']
    del df['close_' + str(26) + '_ema']
    df['ppo'] = ((df['ema_12'] - df['ema_26']) / df['ema_26']) * 100

    del df['ema_12']
    del df['ema_26']

    print_time("Calculation of PPO Done", stime)


def get_ROC(df, col_name, intervals):
    """
    Momentum oscillator
    As per implement https://www.investopedia.com/terms/p/pricerateofchange.asp
    https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    ROC = (close_price_n - close_price_(n-1) )/close_price_(n-1) * 100

    params: df -> dataframe with financial instrument history
            col_name -> column name for which CMO is to be calculated
            intervals -> list of periods for which to calculated

    return: None (adds the result in a column)
    """
    stime = time.time()
    print("Calculating ROC")

    def calculate_roc(series, period):
        return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

    for period in intervals:
        df['roc_' + str(period)] = np.nan
        # for 12 day period, 13th day price - 1st day price
        res = df['close'].rolling(period + 1).apply(calculate_roc, args=(period,), raw=False)
        # print(len(df), len(df[period:]), len(res))
        df['roc_' + str(period)] = res

    print_time("Calculation of ROC done", stime)


# not implemented, can't find
def get_PSI(df, col_name, intervals):
    """
    TODO implement
    """
    pass


def get_DPO(df, col_name, intervals):
    """
    Trend Oscillator type indicator
    """

    stime = time.time()
    print("Calculating DPO")
    for i in tqdm(intervals):
        df['dpo_' + str(i)] = dpo(df['close'], n=i)

    print_time("Calculation of DPO done", stime)


def get_kst(df, col_name, intervals):
    """
    Trend Oscillator type indicator
    """

    stime = time.time()
    print("Calculating KST")
    for i in tqdm(intervals):
        df['kst_' + str(i)] = kst(df['close'], i)

    print_time("Calculation of KST done", stime)


def get_CMF(df, col_name, intervals):
    """
    An oscillator type indicator & volume type
    No other implementation found
    """
    stime = time.time()
    print("Calculating CMF")
    for i in tqdm(intervals):
        df['cmf_' + str(i)] = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], i, fillna=True)

    print_time("Calculation of CMF done", stime)


def get_force_index(df, intervals):
    stime = time.time()
    print("Calculating Force Index")
    for i in tqdm(intervals):
        df['fi_' + str(i)] = force_index(df['close'], df['volume'], 5, fillna=True)

    print_time("Calculation of Force Index done", stime)


def get_EOM(df, col_name, intervals):
    """
    An Oscillator type indicator and volume type
    Ease of Movement : https://www.investopedia.com/terms/e/easeofmovement.asp
    """
    stime = time.time()
    print("Calculating EOM")
    for i in tqdm(intervals):
        df['eom_' + str(i)] = ease_of_movement(df['high'], df['low'], df['volume'], n=i, fillna=True)

    print_time("Calculation of EOM done", stime)


# not used. +1
def get_volume_delta(df):
    stime = time.time()
    print("Calculating volume delta")
    df_ss = sdf.retype(df)
    df_ss['volume_delta']

    print_time("Calculation of Volume Delta done", stime)


# not used. +2 for each interval kdjk and rsv
def get_kdjk_rsv(df, intervals):
    stime = time.time()
    print("Calculating KDJK, RSV")
    df_ss = sdf.retype(df)
    for i in tqdm(intervals):
        df['kdjk_' + str(i)] = df_ss['kdjk_' + str(i)]

    print_time("Calculation of EMA Done", stime)
