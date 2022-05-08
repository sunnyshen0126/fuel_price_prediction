from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
from datetime import datetime
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic
import matplotlib.pyplot as plt


def g_data(last = 0):
    df = pd.read_excel('oil_factor.xlsx')
    df = df[['date', 'Singapore', 'BDI', 'WTI', 'CTFI', 'BDTI', 'BCTI', 'Yield', 'dollar']]
    df.fillna(method='ffill', inplace=True)
    if last == 0:
        return df
    else:
        return df.iloc[:-last]

def check(last = 0):
    df = g_data(last)
    # 获取六个变量的历史数据，在predict_arimax中验证了一阶差分后为平稳序列
    df = df[['date', 'Singapore', 'BDI', 'WTI', 'CTFI', 'BDTI', 'BCTI', 'Yield', 'dollar']]


    df0 = df['date'].iloc[1:]
    df1 = df['Singapore'].diff(1)
    df2 = df['BDI'].diff(1)
    df3 = df['WTI'].diff(1)
    df4 = df['CTFI'].diff(1)
    df5 = df['BDTI'].diff(1)
    df6 = df['BCTI'].diff(1)
    df7 = df['Yield'].diff(1)
    df8 = df['dollar'].diff(1)
    df1.dropna(inplace=True)
    df2.dropna(inplace=True)
    df3.dropna(inplace=True)
    df4.dropna(inplace=True)
    df5.dropna(inplace=True)
    df6.dropna(inplace=True)
    df7.dropna(inplace=True)
    df8.dropna(inplace=True)

    # 协整检验（这里只展示Singapore与其他变量的检验）
    result1 = sm.tsa.stattools.coint(df1, df2)
    result2 = sm.tsa.stattools.coint(df1, df3)
    result3 = sm.tsa.stattools.coint(df1, df4)
    result4 = sm.tsa.stattools.coint(df1, df5)
    result5 = sm.tsa.stattools.coint(df1, df6)
    result6 = sm.tsa.stattools.coint(df1, df7)
    result7 = sm.tsa.stattools.coint(df1, df8)
    #print(result1, result2, result3, result4, result5)

    df7 = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8], axis=1)
    df7.set_index('date', inplace=True)
    return df7


def get_lag(last):
    # 定阶， 比较AIC、BIC等  lag = 5
    df = check(last=last)
    for varLagNum in range(0, 8):
        orgMod = sm.tsa.VARMAX(df, order=(varLagNum, 0), trend='nc', exog=None)
        # 估计：就是模型
        fitMod = orgMod.fit(maxiter=1000, disp=False)
        # 打印统计结果
        print(fitMod.summary())


def cusum():
    df = check()
    orgMod = sm.tsa.VARMAX(df, order=(5, 0), trend='nc', exog=None)
    fitMod = orgMod.fit(maxiter=1000, disp=False)
    resid = fitMod.resid
    #result = {'fitMod':fitMod,'resid':resid}
    result = statsmodels.stats.diagnostic.breaks_cusumolsresid(resid)
    print(result)
    # 绘制脉冲响应图
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    terms = 30
    fitMod.impulse_responses(terms, orthogonalized=True).plot(figsize=(12, 8))
    plt.show()

def variance_decom():
    # 绘制方差分解图
    df = check()
    md = sm.tsa.VAR(df)
    re = md.fit(2)
    fevd = re.fevd(10)
    # 打印出方差分解的结果
    print(fevd.summary())
    # 画图
    fevd.plot(figsize=(4, 16))
    plt.show()


def g_date_N(date, N):
    dates = pd.date_range(start = date, freq = 'D', periods = N)
    dtstr = []
    for i in range(len(dates)):
        dtstr.append(str(dates[i].date()))
    return dtstr


def several_days_prediction_var(day=5, last=0):
    df2 = g_data(last)
    v_list = df2['Singapore'].iloc[1:].tolist()

    df = check()
    df.reset_index(inplace=True)
    start = str(df.loc[len(df) - 1, 'date'])[0:10]
    ms = g_date_N(start, 2)
    day1, day2 = ms[0], ms[1]
    future_days = g_date_N(day2, day)

    data = list()
    date_list = df['date'].tolist()
    a_list = df['Singapore'].tolist()
    b_list = df['BDI'].tolist()
    c_list = df['WTI'].tolist()
    d_list = df['CTFI'].tolist()
    e_list = df['BDTI'].tolist()
    f_list = df['BCTI'].tolist()
    g_list = df['Yield'].tolist()
    h_list = df['dollar'].tolist()

    for i in range(0, len(a_list)):
        row = [a_list[i], b_list[i], c_list[i], d_list[i], e_list[i], f_list[i], g_list[i], h_list[i]]
        data.append(row)
    model = VAR(data)
    model_fit = model.fit()
    #print(model_fit.summary())
    yhat = model_fit.forecast(model_fit.y, steps=day)
    for i in range(0, day):
        v_list.append(yhat[i][0]+v_list[-1])
        date_list.append(future_days[i])
    df1 = pd.DataFrame({'date': date_list, 'predict price': v_list})
    df1.to_excel('short_prediction_var.xlsx')


def five_day_prediction_var(length = 60, start = 60):
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
        df1 = check(last = start-j)

        a_list = df1['Singapore'].tolist()
        b_list = df1['BDI'].tolist()
        c_list = df1['WTI'].tolist()
        d_list = df1['CTFI'].tolist()
        e_list = df1['BDTI'].tolist()
        f_list = df1['BCTI'].tolist()
        g_list = df1['Yield'].tolist()
        h_list = df1['dollar'].tolist()

        data = list()
        for i in range(0, len(a_list)):
            row = [a_list[i], b_list[i], c_list[i], d_list[i], e_list[i], f_list[i], g_list[i], h_list[i]]
            data.append(row)
        model = VAR(data)
        model_fit = model.fit()
        yhat = model_fit.forecast(model_fit.y, steps=5)

        p1_list.append(price_list[-start + j-1] + yhat[0][0])
        p2_list.append(price_list[-start + j-1] + yhat[1][0])
        p3_list.append(price_list[-start + j-1] + yhat[2][0])
        p4_list.append(price_list[-start + j-1] + yhat[3][0])
        p5_list.append(price_list[-start + j-1] + yhat[4][0])
    if length == start:
        d_list = date_list
        p_list = price_list
    else:
        d_list = date_list[:length-start]
        p_list = price_list[:length-start]
    df2 = pd.DataFrame({'date': d_list, 'price': p_list,
                        'p1': p1_list, 'p2': p2_list, 'p3': p3_list, 'p4': p4_list, 'p5': p5_list})

    df2.to_excel('five_day_prediction_var.xlsx')




