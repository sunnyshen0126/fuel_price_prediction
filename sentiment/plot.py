import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def log_return_threshold(factor='f1', mode='train'):
    xy_data = pd.read_excel('../get_data/all.xlsx')
    #df = pd.read_csv('WTI_price.csv')
    #dates = list(set(df['date']))
    #xy_data = xy_data[xy_data['date'].isin(dates)]
    if mode == 'test':
        xy_data = xy_data[(xy_data['date'] >= 20220000)]
        date_range = '  (2022.1~2022.4)  '
    else:
        xy_data = xy_data[(xy_data['date'] <= 20220000) & (xy_data['date'] >= 20190700)]
        date_range = '  (2019.7~2021.12)  '

    xy_data = xy_data.dropna(subset=['r1'])
    xy_data = xy_data[xy_data['r1'] != 0.0]

    threshold_1 = []
    threshold_2 = []
    threshold_3 = []
    threshold_4 = []
    threshold_5 = []

    y_1 = []
    y_3 = []
    y_5 = []
    y_10 = []
    y_20 = []

    for i in range(0, 5):
        low = round(i/5, 2)
        high = round(low+0.2, 2)
        thr_low = xy_data[factor].quantile(low)
        thr_high = xy_data[factor].quantile(high)
        print(thr_low, thr_high)

        y_1_df = xy_data[(xy_data[factor] >= thr_low) & (xy_data[factor] < thr_high)]['r1']
        y_3_df = xy_data[(xy_data[factor] >= thr_low) & (xy_data[factor] < thr_high)]['r3']
        y_5_df = xy_data[(xy_data[factor] >= thr_low) & (xy_data[factor] < thr_high)]['r5']
        y_10_df = xy_data[(xy_data[factor] >= thr_low) & (xy_data[factor] < thr_high)]['r10']
        y_20_df = xy_data[(xy_data[factor] >= thr_low) & (xy_data[factor] < thr_high)]['r20']


        y_1.append(y_1_df.mean())
        y_3.append(y_3_df.mean())
        y_5.append(y_5_df.mean())
        y_10.append(y_10_df.mean())
        y_20.append(y_20_df.mean())


        threshold_1.append(low)
        threshold_2.append(low+0.036)
        threshold_3.append(low+0.072)
        threshold_4.append(low+0.108)
        threshold_5.append(low + 0.144)

    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.bar(x=threshold_1,  height=y_1,  width=0.036, align="edge",  color="green",  edgecolor="grey", label="y_1")
    ax.bar(x=threshold_2,  height=y_3,  width=0.036, align="edge",  color="blue",  edgecolor="grey", label="y_5")
    ax.bar(x=threshold_3,  height=y_5,  width=0.036, align="edge",  color="red",  edgecolor="grey", label="y_10")
    ax.bar(x=threshold_4, height=y_10, width=0.036, align="edge", color="gold", edgecolor="grey", label="y_20")
    ax.bar(x=threshold_5, height=y_20, width=0.036, align="edge", color="black", edgecolor="grey", label="y_60")


    plt.legend(['y_1', 'y_3', 'y_5', 'y_10', 'y_20'])
    ax.set_ylabel('Average return')
    bef = 'Average Log return '
    plt.ylabel('Average log return')#设置y轴标签
    title = bef + factor + date_range
    plt.xlabel('factor')#设置x轴标签
    ax.set_title(title, fontsize=15)

    plt.show()
