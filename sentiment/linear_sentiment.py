import pandas as pd
import jieba.posseg as pseg
from ast import literal_eval
from collections import Counter
import numpy as np


def get_vocab():
    # 获得新闻标题中的词汇
    df = pd.read_excel('../get_data/separate.xlsx')
    sep_list = df['title'].tolist()
    seg_list = []
    count_list = []

    def get_sep(sentence):
        sep = []
        flag = ['ag', 'a', 'ad', 'an', 'vg', 'v', 'vd', 'vn']
        words = pseg.cut(sentence)
        for w in words:
            if w.flag in flag:
                sep.append(w.word)
        return sep
    for i in range(0, len(sep_list)):
        output = get_sep(sep_list[i])
        for j in output:
            if j not in seg_list:
                seg_list.append(j)
                count_list.append(1)
            else:
                ind = seg_list.index(j)
                count_list[ind] += 1

    result = pd.DataFrame({'vocab': seg_list, 'count': count_list})
    result.to_excel('vocab.xlsx')


def calculate_vocab_contribution():
    # 获得不同词对于收益率的贡献
    df1 = pd.read_excel('vocab.xlsx')
    vocab = df1['vocab'].tolist()
    count = np.array([0] * len(vocab))
    ret_1 = np.array([0.0] * len(vocab))
    ret_3 = np.array([0.0] * len(vocab))
    ret_5 = np.array([0.0] * len(vocab))
    ret_10 = np.array([0.0] * len(vocab))
    ret_20 = np.array([0.0] * len(vocab))
    ret_60 = np.array([0.0] * len(vocab))

    df2 = pd.read_excel('../get_data/all.xlsx')
    df2 = df2[df2['date'] <= 20220000]
    df2.dropna(subset=['r1', 'r3', 'r5', 'r10', 'r20', 'r60'], inplace=True)
    content_list = df2['seg'].tolist()
    r1_list = df2['r1'].tolist()
    r3_list = df2['r3'].tolist()
    r5_list = df2['r5'].tolist()
    r10_list = df2['r10'].tolist()
    r20_list = df2['r20'].tolist()
    r60_list = df2['r60'].tolist()

    for i in range(0, len(content_list)):
        if (r1_list[i] != 0) & (len(str(content_list[i])) >= 3):
            saparate = literal_eval(content_list[i])
            result = Counter(saparate)
            length = len(saparate)
            for key, value in result.items():
                if key in vocab:
                    ind = vocab.index(key)
                    count[ind] += value
                    ret_1[ind] += value/length * r1_list[i]
                    ret_3[ind] += value / length * r3_list[i]
                    ret_5[ind] += value/length * r5_list[i]
                    ret_10[ind] += value/length * r10_list[i]
                    ret_20[ind] += value / length * r20_list[i]
                    ret_60[ind] += value / length * r60_list[i]

    df = pd.DataFrame({'vocab': vocab, 'count': count, 'r1': ret_1, 'r3': ret_3, 'r5': ret_5, 'r10': ret_10, 'r20': ret_20, 'r60': ret_60})
    df = df[df['count'] != 0]
    def average_return(ret, count):
        ret = float(ret)
        count = int(count)
        if count == 0:
            return 0
        else:
            return ret/count

    df['r1_average'] = df.apply(lambda x: average_return(x.r1, x['count']), axis=1)
    df['r3_average'] = df.apply(lambda x: average_return(x.r3, x['count']), axis=1)
    df['r5_average'] = df.apply(lambda x: average_return(x.r5, x['count']), axis=1)
    df['r10_average'] = df.apply(lambda x: average_return(x.r10, x['count']), axis=1)
    df['r20_average'] = df.apply(lambda x: average_return(x.r20, x['count']), axis=1)
    df['r60_average'] = df.apply(lambda x: average_return(x.r60, x['count']), axis=1)

    df.to_excel('v_2021.xlsx')


def score1():
    def calculate_score(content, df, type):
        if len(str(content)) >= 3:
            saparate = literal_eval(content)
            result = Counter(saparate)
            vocab = df['vocab'].tolist()
            r1_list = df['r1_average'].tolist()
            r3_list = df['r3_average'].tolist()
            r5_list = df['r5_average'].tolist()
            r10_list = df['r10_average'].tolist()
            r20_list = df['r20_average'].tolist()
            r60_list = df['r60_average'].tolist()
            score0, score1, score2, score3, score4, score5 = 0, 0, 0, 0, 0, 0
            for key, value in result.items():
                if key in vocab:
                    ind = vocab.index(key)
                    score0 += r1_list[ind] * value
                    score1 += r3_list[ind] * value
                    score2 += r5_list[ind] * value
                    score3 += r10_list[ind] * value
                    score4 += r20_list[ind] * value
                    score5 += r60_list[ind] * value
            if type == 1:
                return score1
            elif type == 2:
                return score2
            elif type == 3:
                return score3
            elif type == 4:
                return score4
            elif type == 5:
                return score5
            elif type == 0:
                return score0
        else:
            return 0

    df = pd.read_excel('v_2021.xlsx')
    df2 = pd.read_excel('../get_data/all.xlsx')
    df2['s1'] = df2.apply(lambda x: calculate_score(x.seg, df, 0), axis=1)
    df2['s2'] = df2.apply(lambda x: calculate_score(x.seg, df, 1), axis=1)
    df2['s3'] = df2.apply(lambda x: calculate_score(x.seg, df, 2), axis=1)
    df2['s4'] = df2.apply(lambda x: calculate_score(x.seg, df, 3), axis=1)
    df2['s5'] = df2.apply(lambda x: calculate_score(x.seg, df, 4), axis=1)
    df2['s6'] = df2.apply(lambda x: calculate_score(x.seg, df, 5), axis=1)
    df2.to_excel('../get_data/all.xlsx')




