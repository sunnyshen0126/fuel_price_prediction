from datetime import datetime
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error,mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import TimeseriesGenerator


def g_data(last = 0):
    df = pd.read_excel('oil_factor.xlsx')
    df = df[['date', 'Singapore', 'BDI', 'WTI', 'IMPORT', 'CTFI', 'BDTI', 'BCTI', 'port1', 'port2', 'EXPORT', 'Yield', 'dollar', 'f1', 'f2', 'f3']]
    df.fillna(method='ffill', inplace=True)
    if last == 0:
        return df
    else:
        return df.iloc[:-last]


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    cols, names = [], []
    df = pd.DataFrame(data)
    #n_in, n_in-1, ..., 1，为滞后期数, 分别代表t-n_in, ... ,t-1期
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #0, 1, ..., n_out-1，为超前预测的期数, 分别代表t，t+1， ... ,t+n_out-1期
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def prepare_data(dataset, n_in, n_out=5, n_vars=14, n_train=-5):
    values = dataset.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_in, n_out)
    contain_vars = []
    for i in range(1, n_in + 1):
        contain_vars += [('var%d(t-%d)' % (j, i)) for j in range(1, n_vars + 1)]
    data = reframed[contain_vars + ['var1(t)'] + [('var1(t+%d)' % (j)) for j in range(1, n_out)]]
    col_names = ['Y', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14']

    contain_vars = []
    for i in range(n_vars):
        contain_vars += [('%s(t-%d)' % (col_names[i], j)) for j in range(1, n_in + 1)]
    data.columns = contain_vars + ['Y(t)'] + [('Y(t+%d)' % (j)) for j in range(1, n_out)]
    # 分隔数据集，分为训练集和测试集
    values = data.values
    train = values[:n_train, :]
    test = values[n_train:, :]
    # 分隔输入X和输出y
    train_X, train_y = train[:, :n_in * n_vars], train[:, n_in * n_vars:]
    test_X, test_y = test[:, :n_in * n_vars], test[:, n_in * n_vars:]
    # 将输入X改造为LSTM的输入格式，即[samples,timesteps,features]
    train_X = train_X.reshape((train_X.shape[0], n_in, n_vars))
    test_X = test_X.reshape((test_X.shape[0], n_in, n_vars))
    return scaler, data, train_X, train_y, test_X, test_y, dataset


def fit_lstm(data_prepare, n_neurons=50, n_batch=72, n_epoch=100, loss='mae', optimizer='adam', repeats=1):
    train_X = data_prepare[2]
    train_y = data_prepare[3]
    test_X = data_prepare[4]
    test_y = data_prepare[5]
    model_list = []
    #设计神经网络
    for i in range(repeats):
        model = Sequential()
        model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(train_y.shape[1]))
        model.compile(loss=loss, optimizer=optimizer)
        #拟合神经网络
        history = model.fit(train_X, train_y, epochs=n_epoch, batch_size=n_batch, validation_data=(test_X, test_y), verbose=0, shuffle=False)
        #画出学习过程
        plt.plot(history.history['loss'], color='blue', label='train')
        plt.plot(history.history['val_loss'], color='red',label='test')
        #保存model
        plt.legend(["train","test"])
        #plt.show()
        model_list.append(model)
    return model_list


def lstm_predict(model, data_prepare):
    scaler = data_prepare[0]
    test_X = data_prepare[4]
    test_y = data_prepare[5]
    #做出预测
    yhat = model.predict(test_X)
    #将测试集上的预测值还原为原来的数据维度
    scale_new = MinMaxScaler()
    scale_new.min_, scale_new.scale_ = scaler.min_[0], scaler.scale_[0]
    inv_yhat = scale_new.inverse_transform(yhat)
    #将测试集上的实际值还原为原来的数据维度
    inv_y = scale_new.inverse_transform(test_y)
    return inv_yhat, inv_y


def g_date_N(date, N):
    dates = pd.date_range(start = date, freq = 'D', periods = N)
    dtstr = []
    for i in range(len(dates)):
        dtstr.append(str(dates[i].date()))
    return dtstr


def several_days_prediction_lstm(day=5, n_in=1, n_vars = 14, n_neuron = 5, n_batch = 16, n_epoch = 200, repeats = 10, n_train = 5):
    df_old = g_data()
    start = str(df_old.loc[len(df_old) - 1, 'date'])[0:10]
    ms = g_date_N(start, 2)
    day1, day2 = ms[0], ms[1]
    future_days = g_date_N(day2, day)

    a0_list, a1_list, a2_list, a3_list, a4_list, a5_list, a6_list, a7_list, a8_list, a9_list, a10_list, a11_list, a12_list, a13_list, a14_list, a15_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(0, day):
        a0_list.append(future_days[i])
        a1_list.append(df_old['Singapore'].iloc[-1])
        a2_list.append(df_old['BDI'].iloc[-1])
        a3_list.append(df_old['WTI'].iloc[-1])
        a4_list.append(df_old['IMPORT'].iloc[-1])
        a5_list.append(df_old['CTFI'].iloc[-1])
        a6_list.append(df_old['BDTI'].iloc[-1])
        a7_list.append(df_old['BCTI'].iloc[-1])
        a8_list.append(df_old['port1'].iloc[-1])
        a9_list.append(df_old['port2'].iloc[-1])
        a10_list.append(df_old['EXPORT'].iloc[-1])
        a11_list.append(df_old['Yield'].iloc[-1])
        a12_list.append(df_old['dollar'].iloc[-1])
        a13_list.append(df_old['f1'].iloc[-1])
        a14_list.append(df_old['f2'].iloc[-1])
        a15_list.append(df_old['f3'].iloc[-1])


    df_new = pd.DataFrame({'date': a0_list, 'Singapore': a1_list, 'BDI': a2_list, 'WTI': a3_list, 'IMPORT': a4_list,
                           'CTFI': a5_list, 'BDTI': a6_list, 'BCTI': a7_list, 'port1': a8_list, 'port2': a9_list,
                           'EXPORT': a10_list, 'Yield': a11_list, 'dollar': a12_list, 'f1': a13_list, 'f2': a14_list, 'f3': a15_list})
    df_old = df_old[['date', 'Singapore', 'BDI', 'WTI', 'IMPORT', 'CTFI', 'BDTI', 'BCTI', 'port1', 'port2', 'EXPORT', 'Yield', 'dollar', 'f1', 'f2', 'f3']]
    df = pd.concat([df_old, df_new])
    df.set_index('date', inplace=True)
    data_prepare = prepare_data(dataset=df, n_in=n_in, n_vars = n_vars, n_out=day, n_train=-n_train)
    scaler, data, train_X, train_y, test_X, test_y, dataset = data_prepare
    model = fit_lstm(data_prepare, n_neuron, n_batch, n_epoch, repeats=repeats)
    i1_list = []

    for i in range(repeats):
        i1_list.append(lstm_predict(model[i], data_prepare)[0])

    inv_y = lstm_predict(model[0], data_prepare)[1]

    inv_yhat_ave = 0

    for i in range(repeats):
        inv_yhat_ave += i1_list[i]

    inv_yhat_ave = inv_yhat_ave / repeats

    for i in range(0, day):
        a1_list[i] = inv_yhat_ave[-1][i]
    df2 = pd.DataFrame({'date': a0_list, 'predict price': a1_list})
    df_old = df_old[['date', 'Singapore']]
    df_old.rename(columns={'Singapore': 'predict price'}, inplace=True)
    df3 = pd.concat([df_old, df2])
    df3.to_excel('short_prediction_lstm.xlsx')



def five_day_lstm(df):

    n_in = 1
    n_out = 5
    n_vars = 12
    n_neuron = 5
    n_batch = 16
    n_epoch = 200
    repeats = 5
    n_train = 3
    data_prepare = prepare_data(dataset=df, n_in=n_in, n_vars = n_vars, n_out=n_out, n_train=-n_train)
    scaler, data, train_X, train_y, test_X, test_y, dataset = data_prepare
    model = fit_lstm(data_prepare, n_neuron, n_batch, n_epoch, repeats=repeats)
    i1_list = []

    for i in range(repeats):
        i1_list.append(lstm_predict(model[i], data_prepare)[0])

    inv_y = lstm_predict(model[0], data_prepare)[1]

    inv_yhat_ave = 0

    for i in range(repeats):
        inv_yhat_ave += i1_list[i]

    inv_yhat_ave = inv_yhat_ave / repeats
    return inv_yhat_ave[-1]



def five_day_predcition_lstm(length = 30, start = 30):
    # length为预测的天数（对length天做五日后预测， start为当前时间段多少时间之前）

    df_old = g_data()
    price_list = df_old['Singapore'].tolist()
    date_list = df_old['date'].tolist()

    df_old = df_old[
        ['date', 'Singapore', 'BDI', 'WTI', 'IMPORT', 'CTFI', 'BDTI', 'BCTI', 'port1', 'port2', 'EXPORT', 'Yield',
         'dollar', 'f1', 'f2', 'f3']]
    sta = str(df_old.loc[len(df_old) - 1, 'date'])[0:10]
    ms = g_date_N(sta, 2)
    day1, day2 = ms[0], ms[1]
    future_days = g_date_N(day2, 5)

    a0_list, a1_list, a2_list, a3_list, a4_list, a5_list, a6_list, a7_list, a8_list, a9_list, a10_list, a11_list, a12_list, a13_list, a14_list, a15_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(0, 5):
        a0_list.append(future_days[i])
        a1_list.append(df_old['Singapore'].iloc[-1])
        a2_list.append(df_old['BDI'].iloc[-1])
        a3_list.append(df_old['WTI'].iloc[-1])
        a4_list.append(df_old['IMPORT'].iloc[-1])
        a5_list.append(df_old['CTFI'].iloc[-1])
        a6_list.append(df_old['BDTI'].iloc[-1])
        a7_list.append(df_old['BCTI'].iloc[-1])
        a8_list.append(df_old['port1'].iloc[-1])
        a9_list.append(df_old['port2'].iloc[-1])
        a10_list.append(df_old['EXPORT'].iloc[-1])
        a11_list.append(df_old['Yield'].iloc[-1])
        a12_list.append(df_old['dollar'].iloc[-1])
        a13_list.append(df_old['f1'].iloc[-1])
        a14_list.append(df_old['f2'].iloc[-1])
        a15_list.append(df_old['f3'].iloc[-1])

    df_new = pd.DataFrame({'date': a0_list, 'Singapore': a1_list, 'BDI': a2_list, 'WTI': a3_list, 'IMPORT': a4_list,
                           'CTFI': a5_list, 'BDTI': a6_list, 'BCTI': a7_list, 'port1': a8_list, 'port2': a9_list,
                           'EXPORT': a10_list, 'Yield': a11_list, 'dollar': a12_list, 'f1': a13_list, 'f2': a14_list, 'f3': a15_list})
    df = pd.concat([df_old, df_new])
    df = df[['date', 'Singapore', 'BDI', 'WTI', 'IMPORT', 'CTFI', 'BDTI', 'BCTI', 'port1', 'port2', 'EXPORT', 'Yield',
         'dollar', 'f1', 'f2', 'f3']]
    df.set_index('date', inplace=True)

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
        predict = five_day_lstm(df1)
        p1_list.append(predict[0])
        p2_list.append(predict[1])
        p3_list.append(predict[2])
        p4_list.append(predict[3])
        p5_list.append(predict[4])

    df2 = pd.DataFrame({'date': date_list, 'price': price_list, 'p1': p1_list, 'p2': p2_list, 'p3': p3_list, 'p4': p4_list, 'p5': p5_list})
    df2.to_excel('five_day_prediction_lstm.xlsx')

