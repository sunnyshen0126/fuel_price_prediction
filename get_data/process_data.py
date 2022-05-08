
import pandas as pd
import math
import numpy as np

def get_price_return():
    # 获取燃油价格的历史涨跌幅（主要作用是作为标注）
    df1 = pd.read_excel('/Users/creation/PycharmProjects/scrapy/policy_scrapy/oil_price.xlsx')
    df1 = df1[['指标名称', '现货价(中间价):船用油(0.5%低硫燃料油):新加坡']]
    df1.rename(columns={'指标名称': 'date', '现货价(中间价):船用油(0.5%低硫燃料油):新加坡': 'Singapore'}, inplace=True)
    df1.dropna(subset=['Singapore'], inplace=True)
    df1['after1'] = df1['Singapore'].shift(-1)
    df1['after3'] = df1['Singapore'].shift(-3)
    df1['after5'] = df1['Singapore'].shift(-5)
    df1['after10'] = df1['Singapore'].shift(-10)
    df1['after20'] = df1['Singapore'].shift(-20)
    df1['after60'] = df1['Singapore'].shift(-60)

    def log(x):
        if type(x) == float:
            if x > 0:
                return math.log(x)
            else:
                return np.nan
        else:
            return np.nan

    df1['r1'] = df1.apply(lambda x: log(x.after1 / x.Singapore), axis=1)
    df1['r3'] = df1.apply(lambda x: log(x.after3 / x.Singapore), axis=1)
    df1['r5'] = df1.apply(lambda x: log(x.after5 / x.Singapore), axis=1)
    df1['r10'] = df1.apply(lambda x: log(x.after10 / x.Singapore), axis=1)
    df1['r20'] = df1.apply(lambda x: log(x.after20 / x.Singapore), axis=1)
    df1['r60'] = df1.apply(lambda x: log(x.after60 / x.Singapore), axis=1)

    df1['date'] = df1.apply(lambda x: x.date.strftime("%Y-%m-%d"), axis=1)
    df1['date'] = df1.apply(lambda x: x.date[:4] + x.date[5:7] + x.date[8:10], axis=1)
    #df1 = df1[df1['date'] <= '20220418']

    df1.to_excel('Singapore.xlsx')


def get_close_date():
    # 有些新闻当日并非交易日，则用未来价格作为标注，计算未来若干日的收益率
    df1 = pd.read_excel('separate.xlsx')
    df1['date'] = df1.apply(lambda x: x.date[:4] + x.date[5:7] + x.date[8:10], axis=1)
    df1 = df1[df1['date'] <= '20220418']
    d1_list = df1['date'].tolist()
    title_list = df1['title'].tolist()
    separate_list = df1['separate'].tolist()
    content_list = df1['content'].tolist()
    seg_list = df1['sep'].tolist()
    df2 = pd.read_excel('Singapore.xlsx')
    d2_list = df2['date'].tolist()
    price_list = df2['Singapore'].tolist()
    p1_list = df2['after1'].tolist()
    p3_list = df2['after3'].tolist()
    p5_list = df2['after5'].tolist()
    p10_list = df2['after10'].tolist()
    p20_list = df2['after20'].tolist()
    p60_list = df2['after60'].tolist()
    r1_list = df2['r1'].tolist()
    r3_list = df2['r3'].tolist()
    r5_list = df2['r5'].tolist()
    r10_list = df2['r10'].tolist()
    r20_list = df2['r20'].tolist()
    r60_list = df2['r60'].tolist()

    date1_list = list(set(df1['date']))
    date2_list = list(set(df2['date']))
    date1_list.sort()
    date2_list.sort()

    dd_list = []
    for i in range(0, len(date1_list)):
        #print(date1_list[i])
        first = 0
        for j in range(0, len(date2_list)):

            if (int(date2_list[j]) >= int(date1_list[i])) & (first==0):

                dd_list.append(date2_list[j])
                first += 1

    date_dict = dict(zip(date1_list, dd_list))

    P0_list = []
    P1_list = []
    P3_list = []
    P5_list = []
    P10_list = []
    P20_list = []
    P60_list = []
    R1_list = []
    R3_list = []
    R5_list = []
    R10_list = []
    R20_list = []
    R60_list = []


    for i in range(0, len(d1_list)):
        date = date_dict[d1_list[i]]
        ind = d2_list.index(date)
        P0_list.append(price_list[ind])
        P1_list.append(p1_list[ind])
        P3_list.append(p3_list[ind])
        P5_list.append(p5_list[ind])
        P10_list.append(p10_list[ind])
        P20_list.append(p20_list[ind])
        P60_list.append(p60_list[ind])

        R1_list.append(r1_list[ind])
        R3_list.append(r3_list[ind])
        R5_list.append(r5_list[ind])
        R10_list.append(r10_list[ind])
        R20_list.append(r20_list[ind])
        R60_list.append(r60_list[ind])



    out = pd.DataFrame({'date': d1_list, 'title': title_list, 'separate': separate_list, 'seg': seg_list, 'content': content_list,
                        'price': P0_list, 'p1': P1_list, 'p3':P3_list, 'p5': P5_list, 'p10': P10_list, 'p20': P20_list, 'p60': P60_list,
                        'r1': R1_list, 'r3': R3_list, 'r5': R5_list, 'r10': R10_list, 'r20': R20_list, 'r60': R60_list})
    out.sort_values(by='date', inplace=True)
    out.to_excel('all.xlsx')