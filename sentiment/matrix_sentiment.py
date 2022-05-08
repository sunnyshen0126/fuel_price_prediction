import pandas as pd
from ast import literal_eval
import numpy as np

def calculate_matrix():

    df1 = pd.read_excel('v_2021.xlsx')
    df1 = df1[df1['count'] >= 2] #选择太小的值会导致memory error
    vocab_list = df1['vocab'].tolist()
    count_list = df1['count'].tolist()

    df = pd.read_excel('../get_data/all.xlsx')
    df = df[df['date'] <= 20220000]
    saparate_list = df['separate'].tolist()
    r1_list = df['r1'].tolist()
    r3_list = df['r3'].tolist()
    r5_list = df['r5'].tolist()
    r10_list = df['r10'].tolist()
    r20_list = df['r20'].tolist()
    r60_list = df['r60'].tolist()


    parameter = []  # n * m矩阵，n是vocab数，m是新闻数
    for i in range(0, len(saparate_list)):
        para = []
        if len(str(saparate_list[i])) >= 3:
            saparate = literal_eval(saparate_list[i])
            for j in range(0, len(vocab_list)):
                para.append(saparate.count(vocab_list[j]))
        else:
            for j in range(0, len(vocab_list)):
                para.append(0)
            r1_list[i] = 0
            r3_list[i] = 0
            r5_list[i] = 0
            r10_list[i] = 0
            r20_list[i] = 0
            r60_list[i] = 0

        parameter.append(para)

    P = np.mat(parameter)
    P_1 = (P.T).I
    R1= np.mat(r1_list)
    R3 = np.mat(r3_list)
    R5 = np.mat(r5_list)
    R10 = np.mat(r10_list)
    R20 = np.mat(r20_list)
    R60 = np.mat(r60_list)


    r1_average = np.dot(R1, P_1)
    r1_average = np.array(r1_average)
    r3_average = np.dot(R3, P_1)
    r3_average = np.array(r3_average)
    r5_average = np.dot(R5, P_1)
    r5_average = np.array(r5_average)
    r10_average = np.dot(R10, P_1)
    r10_average = np.array(r10_average)
    r20_average = np.dot(R20, P_1)
    r20_average = np.array(r20_average)
    r60_average = np.dot(R60, P_1)
    r60_average = np.array(r60_average)


    vocab = pd.DataFrame({'vocab': vocab_list, 'count': count_list, 'r1': r1_average[0], 'r3': r3_average[0],
                          'r5': r5_average[0], 'r10': r10_average[0], 'r20': r20_average[0], 'r60': r60_average[0]})
    vocab.to_excel('vocab2.xlsx')



def matrix_score():
    def score_2(content, score_dict):
        score = 0
        if len(str(content)) >= 5:
            saparate = literal_eval(content)
            for j in score_dict.keys():
                score += score_dict[j] * saparate.count(j)
            return score
        else:
            return score


    df = pd.read_excel('../get_data/all.xlsx')
    df1 = pd.read_excel('vocab2.xlsx')
    vocab_list = df1['vocab'].tolist()
    r1_list = df1['r1'].tolist()
    r3_list = df1['r3'].tolist()
    r5_list = df1['r5'].tolist()
    r10_list = df1['r10'].tolist()
    r20_list = df1['r20'].tolist()
    r60_list = df1['r60'].tolist()
    score_dict_1 = dict(zip(vocab_list, r1_list))
    score_dict_2 = dict(zip(vocab_list, r3_list))
    score_dict_3 = dict(zip(vocab_list, r5_list))
    score_dict_4 = dict(zip(vocab_list, r10_list))
    score_dict_5 = dict(zip(vocab_list, r20_list))
    score_dict_6 = dict(zip(vocab_list, r60_list))

    df['m1'] = df.apply(lambda x: score_2(x.separate, score_dict_1), axis=1)
    df['m2'] = df.apply(lambda x: score_2(x.separate, score_dict_2), axis=1)
    df['m3'] = df.apply(lambda x: score_2(x.separate, score_dict_3), axis=1)
    df['m4'] = df.apply(lambda x: score_2(x.separate, score_dict_4), axis=1)
    df['m5'] = df.apply(lambda x: score_2(x.separate, score_dict_5), axis=1)
    df['m6'] = df.apply(lambda x: score_2(x.separate, score_dict_6), axis=1)

    df.to_excel('../get_data/all.xlsx')

