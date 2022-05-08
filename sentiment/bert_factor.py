import pandas as pd
import numpy as np
from datetime import datetime

def to_allfile(directory, name):
    # 将bert训练得到的情绪因子加入到all.xlsx中，情绪因子 = positive - negative（范围为（-1，1））
    file = '../bert/'+directory+'/test_results.tsv'
    df1 = pd.read_csv(file, delimiter='\t')

    a = df1.columns.values
    neg = df1[a[0]].tolist()
    pos = df1[a[1]].tolist()
    factor = np.array([0.0]*(len(df1)+1))
    factor[0] = float(a[1]) - float(a[0])
    for i in range(1, len(factor)):
        factor[i] = pos[i-1] - neg[i-1]
    df = pd.DataFrame({name: factor})
    df1 = pd.read_excel('../get_data/all.xlsx')
    df2 = pd.merge(df1, df, left_index=True, right_index=True)
    df2.to_excel('../get_data/all.xlsx')


def all_factor_to_allfile():
    to_allfile('data_r1', 'f1')
    to_allfile('data_r3', 'f2')
    to_allfile('data_r5', 'f3')
    to_allfile('data_r10', 'f4')
    to_allfile('data_r20', 'f5')
    to_allfile('data_r60', 'f6')


def get_trade_date():
    df1 = pd.read_excel('../get_data/all.xlsx')
    df1 = df1[df1['date'] <= 20220418]
    df2 = pd.read_excel('../get_data/Singapore.xlsx')

    date1_list = list(set(df1['date']))
    date2_list = list(set(df2['date']))
    date1_list.sort()
    date2_list.sort()

    dd_list = []
    for i in range(0, len(date1_list)):
        first = 0
        for j in range(0, len(date2_list)):

            if (int(date2_list[j]) >= int(date1_list[i])) & (first==0):

                dd_list.append(date2_list[j])
                first += 1

    date_dict = dict(zip(date1_list, dd_list))

    df1['date'] = df1.apply(lambda x: date_dict[x.date], axis=1)
    df1['date'] = df1.apply(lambda x: datetime.strptime(str(x.date), '%Y%m%d'), axis=1)
    cols = [col for col in df1.columns if col not in ['Unnamed: 0', 'Unnamed: 0.1', 'date', 'title',
                                                      'separate', 'seg', 'content']]
    df_mean = df1.groupby('date')[cols].mean()
    df_mean.to_excel('all_factor.xlsx')

