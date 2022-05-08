import pandas as pd
from datetime import datetime

def combine_data_factor():
    s_date = datetime.strptime('20190701', '%Y%m%d')
    df1 = pd.read_excel('../oil_price.xlsx')
    df1 = df1[df1['指标名称'] >= s_date]
    df1.rename(columns={'指标名称': 'date', '现货价(中间价):船用油(0.5%低硫燃料油):新加坡': 'Singapore',
                           '波罗的海干散货指数(BDI)': 'BDI', '现货价:原油:美国西德克萨斯中级轻质原油(WTI)': 'WTI',
                           '进口数量:原油:当月值': 'IMPORT', 'CTFI:综合指数': 'CTFI', '原油运输指数(BDTI)': 'BDTI',
                           '成品油运输指数(BCTI)': 'BCTI', '沿海主要港口货物吞吐量:当月值': 'port1', '全国主要港口:集装箱吞吐量:当月值': 'port2',
                           '外贸货物吞吐量:当月值': 'EXPORT', '美元指数': 'dollar', '美国:国债收益率:10年': 'Yield'}, inplace=True)
    df2 = pd.read_excel('../sentiment/all_factor.xlsx')
    df2 = df2[['date', 'f1', 'f2', 'f3']]
    df = pd.merge(df1, df2, on='date')
    df.reset_index(inplace=True)
    df.to_excel('oil_factor.xlsx')
