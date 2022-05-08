import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
import numpy as np
import pyflux as pf

def g_data(last=0):
    df = pd.read_excel('../oil_price.xlsx')
    df1 = df[['指标名称', '现货价(中间价):船用油(0.5%低硫燃料油):新加坡']]
    df1.rename(columns = {'指标名称': 'date', '现货价(中间价):船用油(0.5%低硫燃料油):新加坡': 'price'}, inplace=True)
    df1.dropna(subset=['price'], inplace=True)
    if last == 0:
        return df1
    else:
        return df1.iloc[:-last]

from statsmodels.tsa.stattools import adfuller

def get_difference():
    df = g_data()
    no_diff = adfuller(df.price)

    print('The ADF Statistic of original Singapore price %f' % no_diff[0])

    print('The p value of original Singapore price: %f' % no_diff[1])

    df1 = df['price'].diff(1)
    df1.dropna(inplace=True)
    first_diff = adfuller(df1)

    print('The ADF Statistic of 1st difference Singapore price: %f' % first_diff[0])

    print('The p value of 1st difference Singapore price: %f' % first_diff[1])


def get_correlation_plot():
    df1 = g_data()
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    diff1 = df1['price'].diff(2)
    diff1.dropna(inplace=True)
    fig = sm.graphics.tsa.plot_acf(diff1, lags=50,ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff1, lags=50,ax=ax2)
    plt.show()


# q=4, p=7
# 过去一年的每日滚动预测
def one_day_prediction_arima(day=250, last=0):
    df1 = g_data(last)
    result = []
    date = []
    for i in range(0, day):
        df2 = df1.iloc[:-day+i]
        model = ARIMA(df2['price'], order=(7, 1, 4))
        fitted = model.fit(disp=0)
        fc, se, conf = fitted.forecast(steps=1)
        date.append(df2.iloc[-1]['date'])
        result.append(fc[0])
    df3 = pd.DataFrame({'date': date, 'predict': result})
    df3.to_excel('ARIMA_day.xlsx')


def g_date_N(date, N):
    dates = pd.date_range(start = date, freq = 'D', periods = N)
    dtstr = []
    for i in range(len(dates)):
        dtstr.append(str(dates[i].date()))
    return dtstr


def several_days_prediction_arima(day=5, last=0):
    df = g_data(last)
    d_list = df['date'].tolist()
    v_list = df['price'].tolist()
    df.reset_index(inplace=True)
    start = str(df.loc[len(df) - 1, 'date'])[0:10]
    ms = g_date_N(start, 2)
    day1, day2 = ms[0], ms[1]
    future_days = g_date_N(day2, day)
    model = ARIMA(df['price'], order=(7, 1, 4))
    fitted = model.fit(disp=0)
    fc, se, conf = fitted.forecast(steps=day)
    for i in range(0, day):
        d_list.append(future_days[i])
        v_list.append(fc[i])
    df1 = pd.DataFrame({'date': d_list, 'price': v_list})
    df1.to_excel('short_prediction_arima.xlsx')
