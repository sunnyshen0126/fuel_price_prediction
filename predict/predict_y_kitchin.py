import pandas as pd
import numpy as np
import datetime
import math
import warnings
warnings.filterwarnings("ignore")

#PREDICT_LEN = 12 #PREDICT_LEN为预测周期的长度

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


def g_date_N(date, N):
    dates = pd.date_range(start = date, freq = 'M', periods = N)
    dtstr = []
    for i in range(len(dates)):
        dtstr.append(str(dates[i].date()))
    return dtstr


def get_prediction(PREDICT_LEN_1, last=0):

     code = '现货价:原油:美国西德克萨斯中级轻质原油(WTI)'
     df = g_data(last)
     start = str(df.loc[len(df) - 1, 'ddate'])[0:10]
     ms = g_date_N(start, 2)
     month1, month2= ms[0], ms[1]
     ###########_________Year on year series____________________
     for i in range(PREDICT_LEN_1,len(df)):
         df.loc[i,'对数同比'] = math.log(df.loc[i,'value']) - math.log(df.loc[i-PREDICT_LEN_1,'value'])
     data = df[PREDICT_LEN_1::]
     data = data.reset_index(drop=True)
     seq = list(data['对数同比'])
     ###################_______Calculate period, phase and amplitude_______________________
     N = len(seq)
     fs = 1
     NFFT = 2**(math.ceil(math.log(N, 2))+4)    #subdivide the data, and then you can get more accurate and detailed peaks
     Y = np.fft.fft(seq, NFFT)/N #It's about subdividing the data, and then you can get more accurate and detailed peak situation. Gaussian filter has a parameter, which is the K power of 2. The above line calculates the K, Fourier transform
     f = fs/2*np.linspace(1e-9, 1, int(NFFT/2+1))
     T = 1/f
     YY=Y[:len(T)]

     tab = pd.DataFrame(columns=['周期', '振幅', 'value'])
     tab['周期'] = T
     tab['振幅'] = abs(YY)
     tab['value'] = (abs(YY) * N)**2

     tab0=tab[tab['周期']<150]
     tab0=tab0[tab0['周期']>15]
     tab0=tab0.reset_index(drop=True)

     from scipy.signal import find_peaks  #find peak

     def get_T(tab0,k):
         peaks = find_peaks(tab0['value'])[0]
         tab00 = pd.DataFrame(columns=['周期','value'])
         tab00['周期'] = tab0['周期'][peaks]
         tab00['value'] = tab0['value'][peaks]
         tab00 = tab00.sort_values(by = 'value',axis = 0,ascending = False)
         tab00 = tab00.reset_index(drop=True)
         # print(tab00)
         return tab00['周期'][:k]
     month =  get_T(tab0,2).values
     month = month.tolist()
     # print(month)

     from suffix import new_fft
     future_len = PREDICT_LEN_1
     outs = new_fft(seq,l = 5,month = month,t = future_len)

     future_id = list(range(-1,-1-future_len,-1))
     future_months = g_date_N(month2, future_len)
     for id in range(future_len):
         data.loc[future_id[id],:] = [future_months[id],0,0]

     data = data.reset_index(drop=True)
     data['sin_V'] = np.real(outs)
     data['周期1'] = [month[0]]*len(outs)
     if len(month) == 1:
        data['周期2'] = [None]*len(outs)
     else:
        data['周期2'] = [month[1]]*len(outs)
     data['code']= [code]*len(outs)
     data['name']= [code]*len(outs)

     tab = data[0:len(data)-future_len]
     import statsmodels.api as sm
     yy = tab.loc[:,'对数同比']
     sinv = tab.loc[:,'sin_V']
     X = sm.add_constant(sinv)  # Adding a column of constants 1 before x is convenient for regression with intercept term
     model = sm.OLS(yy, X)
     results = model.fit()
     data['模拟'] = results.params[1] * data['sin_V'] + results.params[0]
     ####__________________________________
     data.loc[len(data)-future_len:len(data)-1,'对数同比'] = list(data.loc[len(data)-future_len:len(data)-1,'模拟'])

     for le in range(future_len):
         data.loc[len(data) - future_len + le, 'value'] = \
             data.loc[len(data) - future_len - 1, 'value'] * math.exp(data.loc[len(data) - future_len + le, '对数同比'])


     data['ddate'] = pd.to_datetime(data['ddate'])
     data['日期'] = data['ddate']
     data['名称'] = data['name']
     data = data[['日期', '名称', 'value', 'sin_V']]
     return data['日期'].iloc[-1], data['value'].iloc[-1]


def predict_kitchin(predict_len=12):
    df = g_data()
    d_list = df['ddate'].tolist()
    v_list = df['value'].tolist()
    for i in range(1, predict_len+1):
        date, value = get_prediction(i)
        d_list.append(date)
        v_list.append(value)
    df1 = pd.DataFrame({'date': d_list, 'value': v_list})
    df1.to_excel('long_prediction_kitchin.xlsx')


def predict_year_kitchin(last=0):
    # 对未来一年做未来十二个月的预测
    df = g_data(last)
    d_list = df['ddate'].tolist()
    v_list = df['value'].tolist()

    p1_list, p2_list, p3_list, p4_list, p5_list, p6_list, p7_list, p8_list, p9_list, p10_list, p11_list, p12_list = [], [], [], [], [], [], [], [], [], [], [], []
    for j in range(0, len(v_list)-12):
        p1_list.append(v_list[j])
        p2_list.append(v_list[j])
        p3_list.append(v_list[j])
        p4_list.append(v_list[j])
        p5_list.append(v_list[j])
        p6_list.append(v_list[j])
        p7_list.append(v_list[j])
        p8_list.append(v_list[j])
        p9_list.append(v_list[j])
        p10_list.append(v_list[j])
        p11_list.append(v_list[j])
        p12_list.append(v_list[j])

    for i in range(0, 12):
        print(i)
        last = 11-i
        date1, value1 = get_prediction(PREDICT_LEN_1=1, last=last)
        date2, value2 = get_prediction(PREDICT_LEN_1=2, last=last)
        date3, value3 = get_prediction(PREDICT_LEN_1=3, last=last)
        date4, value4 = get_prediction(PREDICT_LEN_1=4, last=last)
        date5, value5 = get_prediction(PREDICT_LEN_1=5, last=last)
        date6, value6 = get_prediction(PREDICT_LEN_1=6, last=last)
        date7, value7 = get_prediction(PREDICT_LEN_1=7, last=last)
        date8, value8 = get_prediction(PREDICT_LEN_1=8, last=last)
        date9, value9 = get_prediction(PREDICT_LEN_1=9, last=last)
        date10, value10 = get_prediction(PREDICT_LEN_1=10, last=last)
        date11, value11 = get_prediction(PREDICT_LEN_1=11, last=last)
        date12, value12 = get_prediction(PREDICT_LEN_1=12, last=last)
        p1_list.append(value1)
        p2_list.append(value2)
        p3_list.append(value3)
        p4_list.append(value4)
        p5_list.append(value5)
        p6_list.append(value6)
        p7_list.append(value7)
        p8_list.append(value8)
        p9_list.append(value9)
        p10_list.append(value10)
        p11_list.append(value11)
        p12_list.append(value12)
    df2 = pd.DataFrame({'date': d_list, 'price': v_list, 'p1': p1_list, 'p2': p2_list,
                        'p3': p3_list, 'p4': p4_list, 'p5': p5_list, 'p6': p6_list,
                        'p7': p7_list, 'p8': p8_list, 'p9': p9_list, 'p10': p10_list,
                        'p11': p11_list, 'p12': p12_list})

    df2.to_excel('year_prediction_kitchin.xlsx')

