import pandas as pd
import math
from datetime import datetime

def get_year_Singapore(last=0):
    # 首先按照WTI原油价格的涨跌幅得到燃油的预测价格
    rets = pd.read_excel('../oil_price.xlsx')
    rets = rets[['指标名称', '现货价(中间价):船用油(0.5%低硫燃料油):新加坡']]
    rets.rename(columns={'指标名称': 'ddate', '现货价(中间价):船用油(0.5%低硫燃料油):新加坡': 'Singapore'}, inplace=True)
    rets.dropna(subset=['Singapore'], inplace=True)

    rets['ddate'] = pd.to_datetime(rets['ddate'])
    rets = rets.set_index("ddate")
    rets = rets.resample('M').last()
    date_list = list(rets.index)[-12:]
    value_list = list(rets['Singapore'])[-12:]
    df1 = pd.read_excel('../predict/year_prediction_kitchin.xlsx')
    p1_list, p2_list, p3_list = [], [], []
    a_list = df1['price'].iloc[-12:].tolist()
    b_list = df1['p1'].iloc[-12:].tolist()
    c_list = df1['p2'].iloc[-12:].tolist()
    d_list = df1['p3'].iloc[-12:].tolist()
    for i in range(0, 12):
        p1_list.append(value_list[i] * b_list[i] / a_list[i])
        p2_list.append(value_list[i] * c_list[i] / a_list[i])
        p3_list.append(value_list[i] * d_list[i] / a_list[i])
    df = pd.DataFrame({'date': date_list, 'price': value_list, 'p1': p1_list, 'p2': p2_list, 'p3': p3_list})
    if last == 0:
        return df
    else:
        return df.iloc[:-last]


def year_buy_signal(last=0, storage=5, amount=10000): # 假设一个月的储存成本为5美元，需要采购10000吨
    df = get_year_Singapore(last)
    date_list = df['date'].tolist()
    price_list = df['price'].tolist()
    p1_list = df['p1'].tolist()
    p2_list = df['p2'].tolist()
    p3_list = df['p3'].tolist()
    buy_list = []
    average_price = 0
    average_buy = 0
    total = amount
    for i in range(0, len(date_list)):
        average_price += price_list[i]
        if price_list[i] < min(p1_list[i] - storage, p2_list[i] - 2*storage, p3_list[i] - 3*storage):
            buy = min(4*amount/(12-i), amount)
            amount = amount - buy
            buy_list.append(buy)
        else:
            buy = 0
            buy_list.append(buy)
        average_buy += buy * price_list[i]
    average_price = average_price/12
    average_buy = average_buy/total
    print('Average price: ', average_price, '  ', 'Average purchase price: ', average_buy)
    df1 = pd.DataFrame({'date': date_list, 'price': price_list, 'buy amount': buy_list})
    df1.to_excel('year_buy_signal.xlsx')


def month_buy_signal(start='20220301', end='20220331', storage = 0.2): # 假设每日储存成本为0.2美元
    # 选定测试月份（一个月）
    df = pd.read_excel('../predict/five_day_prediction.xlsx') # 这里可以取任意一种预测方式的结果（这里选取的是bilstm的预测结果）
    s_date = datetime.strptime(start, '%Y%m%d')
    e_date = datetime.strptime(end, '%Y%m%d')
    df1 = df[(df['date'] >= s_date) & (df['date'] <= e_date)]
    day = len(df1) # 共有day个交易日
    amount = day # 假设需要采购的量即为天数份
    buy_list = []
    average_price = 0
    average_buy = 0
    date_list = df1['date'].tolist()
    price_list = df1['price'].tolist()
    p1_list = df1['p1'].tolist()
    p2_list = df1['p2'].tolist()
    p3_list = df1['p3'].tolist()
    p4_list = df1['p4'].tolist()
    p5_list = df1['p5'].tolist()
    for j in range(0, day):
        average_price += price_list[j]
        if j != day-1:
            if (price_list[j] < min(p1_list[j] - storage, p2_list[j] - storage*2, p3_list[j] - storage*3, p4_list[j] -storage*4, p5_list[j]-storage*5)) & (amount>0):
                average_buy += price_list[j] * min(amount, 6)
                buy_list.append(min(amount, 6))
                amount = max(0, amount-6)
            else:
                buy_list.append(0)
        else:
            if amount > 0:
                buy_list.append(amount)
                average_buy += price_list[j] * amount
                amount = 0
            else:
                buy_list.append(0)

    average_price = average_price / day
    average_buy = average_buy / day
    print('Average price: ', average_price, '  ', 'Average purchase price: ', average_buy)
    df2 = pd.DataFrame({'date': date_list, 'price': price_list, 'buy amount': buy_list})
    df2.to_excel('month_buy_signal.xlsx')


def week_buy_signal(start = 40):
    # 从过去start天开始采购计划（必须要有预测）,由于需要整周，因此start取5的倍数
    df = pd.read_excel('../predict/five_day_prediction_bilstm.xlsx') # 这里可以取任意一种预测方式的结果（这里选取的是bilstm的预测结果）
    df1 = df.iloc[-start:]
    date_list = df1['date'].tolist()
    price_list = df1['price'].tolist()
    split = int(start / 5) # 这里假设五天为一周（但实际情况有些周仅四个或更少交易日）
    buy_list = []
    average_price = 0
    average_buy = 0
    for i in range(0, split):
        bought = 0
        if i != split - 1:
            week = df1.iloc[5*i: 5*i+5]
        else:
            week = df1.iloc[5*i:]
        p_list = week['price'].tolist()
        p1_list = week['p1'].tolist()
        p2_list = week['p2'].tolist()
        p3_list = week['p3'].tolist()
        p4_list = week['p4'].tolist()
        predict_list = []
        predict_list.append(min(p1_list[0], p2_list[0], p3_list[0], p4_list[0]))
        predict_list.append(min(p1_list[1], p2_list[1], p3_list[1]))
        predict_list.append(min(p1_list[2], p3_list[2]))
        predict_list.append(p1_list[3])
        predict_list.append(p_list[-1])
        for j in range(0, 5):
            average_price += p_list[j]
            if (p_list[j] <= predict_list[j]) & (bought == 0):
                buy_list.append(1)
                average_buy += predict_list[j]
                bought = 1
            else:
                buy_list.append(0)

    average_price = average_price/start
    average_buy = average_buy/split
    print('Average price: ', average_price, '  ', 'Average purchase price: ', average_buy)
    df2 = pd.DataFrame({'date': date_list, 'price': price_list, 'buy amount': buy_list})
    df2.to_excel('week_buy_signal.xlsx')

