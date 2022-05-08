import pandas as pd
import numpy as np
import datetime
import math
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

def g_data(last=0):
    # rets = BST.sql(schema='dm_macro').read('dm_quant_macro_edb_chinamacro', select=['TRADE_DT', 'VALUE', 'INDEX_NAME'], \
    #                                 condD={'in': {'INDEX_NAME': ['现货价:原油:美国西德克萨斯中级轻质原油(WTI)']}})\
    #             .drop_duplicates(['TRADE_DT', 'INDEX_NAME'])#.pivot(index='日期', columns='指标名称', values='数值')
    rets = pd.read_excel('../oil_price.xlsx')
    rets = rets[['指标名称', '现货价:原油:美国西德克萨斯中级轻质原油(WTI)']]
    rets.rename(columns={'指标名称': 'ddate', '现货价:原油:美国西德克萨斯中级轻质原油(WTI)': 'WTI'}, inplace=True)
    rets.dropna(subset=['WTI'], inplace=True)

    rets['ddate'] = pd.to_datetime(rets['ddate'])
    rets = rets.set_index("ddate")
    rets = rets.resample('M').last()
    df = pd.DataFrame()
    df['ddate'] = list(rets.index)
    df['value'] = list(rets['WTI'])
    if last == 0:
        return df
    elif last > 0:
        return df.iloc[:-last]


def check_difference():
    df = g_data()
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    diff1 = df['value'].diff(1)
    # d = 1
    diff1.dropna(inplace=True)
    fig = sm.graphics.tsa.plot_acf(diff1, lags=50, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff1, lags=50, ax=ax2)
    plt.show()

    ax1.plot(df['ddate'], diff1, color='red')
    ax1.set_ylabel('difference')
    ax1.set_xlabel('date')
    plt.title('2nd Difference')
    plt.show()


def g_date_N(date, N):
    dates = pd.date_range(start = date, freq = 'M', periods = N)
    dtstr = []
    for i in range(len(dates)):
        dtstr.append(str(dates[i].date()))
    return dtstr



def predict_arima(PREDICT_LEN_1=12, last=0, alpha=0.05):
    df = g_data(last)
    start = str(df.loc[len(df) - 1, 'ddate'])[0:10]
    ms = g_date_N(start, 2)
    month1, month2 = ms[0], ms[1]
    future_months = g_date_N(month2, PREDICT_LEN_1)

    model = ARIMA(df['value'], order=(2, 1, 2))
    fitted = model.fit(disp=0)
    fc, se, conf = fitted.forecast(steps = PREDICT_LEN_1, alpha=alpha) # 95% conf

    d_list = df['ddate'].tolist()
    v_list = df['value'].tolist()
    h_list = df['value'].tolist()
    l_list = df['value'].tolist()

    for i in range(0, PREDICT_LEN_1):
        d_list.append(future_months[i])
        v_list.append(fc[i])
        l_list.append(conf[i][0])
        h_list.append(conf[i][1])

    data = pd.DataFrame({'date': d_list, 'value': v_list, 'low': l_list, 'high': h_list})
    data.to_excel('long_prediction_arima.xlsx')
