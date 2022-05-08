from datetime import datetime
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
import numpy as np
import pyflux as pf
from sklearn.metrics import mean_absolute_error,mean_squared_error


def g_data(last = 0):
    df = pd.read_excel('oil_factor.xlsx')
    df = df[['date', 'Singapore', 'BDI', 'WTI', 'CTFI', 'BDTI', 'BCTI', 'Yield', 'dollar']]
    df.fillna(method='ffill', inplace=True)
    if last == 0:
        return df
    else:
        return df.iloc[:-last]

from statsmodels.tsa.stattools import adfuller



def get_difference():
    df = g_data()

    S0 = adfuller(df['Singapore'])
    df0 = df['Singapore'].diff(1)
    df0.dropna(inplace=True)
    S1 = adfuller(df0)
    print('The ADF Statistic of original Singapore price %f' % S0[0])
    print('The p value of original Singapore price: %f' % S0[1])
    print('The ADF Statistic of 1st difference Singapore price: %f' % S1[0])
    print('The p value of 1st difference Singapore price: %f' % S1[1])

    B0 = adfuller(df['BDI'])
    df1 = df['BDI'].diff(1)
    df1.dropna(inplace=True)
    B1 = adfuller(df1)
    print('The ADF Statistic of original BDI %f' % B0[0])
    print('The p value of original BDI: %f' % B0[1])
    print('The ADF Statistic of 1st difference BDI: %f' % B1[0])
    print('The p value of 1st difference BDI: %f' % B1[1])

    W0 = adfuller(df['WTI'])
    df2 = df['WTI'].diff(1)
    df2.dropna(inplace=True)
    W1 = adfuller(df2)
    print('The ADF Statistic of original WTI %f' % W0[0])
    print('The p value of original WTI: %f' % W0[1])
    print('The ADF Statistic of 1st difference WTI: %f' % W1[0])
    print('The p value of 1st difference WTI: %f' % W1[1])

    C0 = adfuller(df['CTFI'])
    df3 = df['CTFI'].diff(1)
    df3.dropna(inplace=True)
    C1 = adfuller(df3)
    print('The ADF Statistic of original CTFI %f' % C0[0])
    print('The p value of original CTFI: %f' % C0[1])
    print('The ADF Statistic of 1st difference CTFI: %f' % C1[0])
    print('The p value of 1st difference CTFI: %f' % C1[1])

    D0 = adfuller(df['BDTI'])
    df4 = df['BDTI'].diff(1)
    df4.dropna(inplace=True)
    D1 = adfuller(df4)
    print('The ADF Statistic of original BDTI %f' % D0[0])
    print('The p value of original BDTI: %f' % D0[1])
    print('The ADF Statistic of 1st difference BDTI: %f' % D1[0])
    print('The p value of 1st difference BDTI: %f' % D1[1])

    T0 = adfuller(df['BCTI'])
    df5 = df['BCTI'].diff(1)
    df5.dropna(inplace=True)
    T1 = adfuller(df5)
    print('The ADF Statistic of original BCTI %f' % T0[0])
    print('The p value of original BCTI: %f' % T0[1])
    print('The ADF Statistic of 1st difference BCTI: %f' % T1[0])
    print('The p value of 1st difference BCTI: %f' % T1[1])

    Y0 = adfuller(df['yield'])
    df6 = df['yield'].diff(1)
    df6.dropna(inplace=True)
    Y1 = adfuller(df6)
    print('The ADF Statistic of original 10 year yield %f' % Y0[0])
    print('The p value of original 10 year yield: %f' % Y0[1])
    print('The ADF Statistic of 1st difference 10 year yield: %f' % Y1[0])
    print('The p value of 1st difference 10 year yield: %f' % Y1[1])

    dollar0 = adfuller(df['dollar'])
    df7 = df['dollar'].diff(1)
    df7.dropna(inplace=True)
    dollar1 = adfuller(df7)
    print('The ADF Statistic of original dollar index %f' % dollar0[0])
    print('The p value of original dollar index: %f' % dollar0[1])
    print('The ADF Statistic of 1st difference dollar index: %f' % dollar1[0])
    print('The p value of 1st difference dollar index: %f' % dollar1[1])



def find_ar_ma():
    # 通过迭代找到最合适的p和q
    df = g_data()
    p = np.arange(7)
    q = np.arange(7)
    pp, qq = np.meshgrid(p, q)
    resultdf = pd.DataFrame(data={"arp": pp.flatten(), "mrq": qq.flatten()})
    resultdf["bic"] = np.double(pp.flatten())
    resultdf["mae"] = np.double(qq.flatten())
    traindata = df.iloc[:500]
    testdata = df.iloc[500:]

    ## 迭代循环建立多个模型
    for ii in resultdf.index:
        model_i = pf.ARIMAX(data=traindata, formula="Singapore~BDI+WTI+CTFI+BDTI+BCTI+Yield+dollar", ar=resultdf.arp[ii],
                            ma=resultdf.mrq[ii], integ=1)
        modeli_fit = model_i.fit()
        bic = modeli_fit.bic
        pre = model_i.predict(h=testdata.shape[0], oos_data=testdata)
        a = testdata['Singapore'].diff(1)
        a.fillna(0, inplace=True)
        x = mean_absolute_error(a, pre['Differenced Singapore'])
        resultdf.bic[ii] = bic
        resultdf.mae[ii] = x

    print("模型迭代结束")

    ## 按照BIC寻找合适的模型
    print(resultdf.sort_values(by="bic").head())


def one_day_prediction_arimax(day=30, last=0):

    df = g_data(last)
    price = df['Singapore'].tolist()
    t1 = df.iloc[:-day]
    predict = t1['Singapore'].tolist()
    date = df['date'].tolist()

    for i in range(0, day):
        print(i)
        traindata = df.iloc[:-day+i]
        testdata = df.iloc[-day+i:]
        model = pf.ARIMAX(data=traindata, formula="Singapore~BDI+WTI+CTFI+BDTI+BCTI+Yield+dollar", ar=3, ma=5, integ=1)
        model_1 = model.fit()

        pre = model.predict(h=1, oos_data=testdata)
        predict.append(pre['Differenced Singapore'].iloc[0] + traindata['Singapore'].iloc[-1])


    df1 = pd.DataFrame({'date': date, 'actual price': price, 'predict price': predict})
    df1.to_excel('ARIMAX_day.xlsx')


def g_date_N(date, N):
    dates = pd.date_range(start = date, freq = 'D', periods = N)
    dtstr = []
    for i in range(len(dates)):
        dtstr.append(str(dates[i].date()))
    return dtstr


def several_days_prediction_arimax(day=5, last = 0):
    df = g_data(last)
    d_list = df['date'].tolist()
    v_list = df['Singapore'].tolist()
    df.reset_index(inplace=True)
    start = str(df.loc[len(df) - 1, 'date'])[0:10]
    ms = g_date_N(start, 2)
    day1, day2 = ms[0], ms[1]
    future_days = g_date_N(day2, day)
    testdata = df.iloc[-day:]
    model = pf.ARIMAX(data=df, formula="Singapore~BDI+WTI+CTFI+BDTI+BCTI+dollar+Yield", ar=3, ma=5, integ=1)
    model_1 = model.fit()
    predict = model.predict(h=day, oos_data=testdata)
    for i in range(0, day):
        d_list.append(future_days[i])
        v_list.append(v_list[-1]+predict['Differenced Singapore'].iloc[i])

    df1 = pd.DataFrame({'date': d_list, 'predict price': v_list})
    df1.to_excel('short_prediction_arimax.xlsx')


def five_day_predcition_arimax(length = 60, start = 60):
    # 预测五天，从过去六十天开始预测预测length（60）天 start >= length
    df = g_data()
    date_list = df['date'].tolist()
    price_list = df['Singapore'].tolist()

    p1_list, p2_list, p3_list, p4_list, p5_list = [], [], [], [], []
    for i in range(0, len(price_list) - start):
        p1_list.append(price_list[i])
        p2_list.append(price_list[i])
        p3_list.append(price_list[i])
        p4_list.append(price_list[i])
        p5_list.append(price_list[i])

    for j in range(1, length+1):
        if j == start:
            df1 = df
        else:
            df1 = df.iloc[:-start+j]
        '''
        if start > j:
            testdata = df.iloc[-start+j:]
        else:
            testdata = df.iloc[-1]
        '''
        testdata = df1.iloc[-5:]

        model = pf.ARIMAX(data=df1, formula="Singapore~BDI+WTI+CTFI+BDTI+BCTI+dollar+Yield", ar=3, ma=5, integ=1)
        model_1 = model.fit()
        predict = model.predict(h=5, oos_data=testdata)
        p1_list.append(price_list[-start + j-1] + predict['Differenced Singapore'].iloc[0])
        p2_list.append(price_list[-start + j-1] + predict['Differenced Singapore'].iloc[1])
        p3_list.append(price_list[-start + j-1] + predict['Differenced Singapore'].iloc[2])
        p4_list.append(price_list[-start + j-1] + predict['Differenced Singapore'].iloc[3])
        p5_list.append(price_list[-start + j-1] + predict['Differenced Singapore'].iloc[4])
    if length == start:
        d_list = date_list
        p_list = price_list
    else:
        d_list = date_list[:length-start]
        p_list = price_list[:length-start]
    df2 = pd.DataFrame({'date': d_list, 'price': p_list,
                        'p1': p1_list, 'p2': p2_list, 'p3': p3_list, 'p4': p4_list, 'p5': p5_list})

    df2.to_excel('five_day_prediction_arimax.xlsx')


