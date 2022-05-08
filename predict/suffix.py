
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from scipy.fftpack import fft,ifft, fftshift
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False
import math
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from statsmodels.regression.linear_model import OLS
def fspecial(shape=(3,3),sigma=0.5):
    """
    fspecial - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def nextpow2(n):
    return math.ceil(math.log(n, 2))

def showfft(signal, order):
    N = len(signal)
    fs = 1
    NFFT = 2**(nextpow2(N)+3)
    Y = np.fft.fft(signal, NFFT)
    f = fs/2*np.linspace(0, 1, NFFT)
    T = 1/f
    sig = abs(Y[:len(T)])
    if order:
        plt.plot(T[30:], sig[30:])
        plt.show()
    return T, sig



def new_fft(signal, l, month=[21, 42, 100, 200], t=0):
    if len(signal) <= 24:
        l = 1

    N = len(signal)
    h = fspecial((l, 1), 1)  # np.array([[1,1,1,1,1,1,1]]).T
    fs = 1
    NFFT = int(2 ** (nextpow2(N) + 3))
    # warnings.filterwarnings("ignore")

    Y = np.fft.fft(signal, NFFT)
    f = fs / 2 * np.linspace(0, 1, NFFT // 2 + 1)
    T = 1 / f

    w = np.zeros([len(Y), 1])
    for i in range(len(month)):
        center = np.argmin(abs(f - 1 / month[i]))
        center2 = len(Y) - center - 1
        w[int(center - (l - 1) / 2): int(center + (l - 1) / 2) + 1] += h  # 此处需要更改，需要+
        w[int(center2 - (l - 1) / 2): int(center2 + (l - 1) / 2) + 1] += h
    '''
    w = np.array(w)
    w[w>np.max(h)]=np.max(h)
    '''
    Y2 = np.multiply(Y, w.T[0])

    outs = np.fft.ifft(Y2)
    outs = outs[:(N + t)]
    return outs


def find_peaks_interp(sig, T, max_cycle, isplot):
    x_data = T[T <= max_cycle]
    y_data = sig[T <= max_cycle]
    x_data_smooth = np.linspace(min(x_data), max(x_data), 400000)
    it1 = interp1d(x_data, y_data)
    y_data_smooth = it1(x_data_smooth)
    peaks, property = find_peaks(y_data_smooth, width=4000)
    if isplot:
        plt.figure()
        plt.plot(x_data, y_data)
        plt.scatter(x_data_smooth[peaks], y_data_smooth[peaks])
        plt.show()
    return x_data_smooth[peaks]


def find_period_time(self, log_data, price_data):
    T, sig = showfft(log_data, False)
    sig = sig[::-1]
    T = T[::-1]

    sig = sig[T <= self.max_cycle]
    T = T[T <= self.max_cycle]

    if self.inst_info.period is not None:
        period = self.inst_info.period
    else:
        period = self.find_peaks_interp(sig, T, self.max_cycle, False)

    nowfft = new_fft(log_data, 7, period, 0)
    fowardfft = new_fft(log_data, 7, period, self.prediction_time)

    _signal = self.trans(np.real(nowfft))
    _signal_forward = self.trans(np.real(fowardfft))

    regr = OLS(log_data, _signal)
    results = regr.fit()
    signal = _signal*results.params[0]
    signal_forward = _signal_forward*results.params[0]

    signal_forward_dlog = self.dsamp_t(price_data, signal_forward, log_data)

    return signal_forward, signal_forward_dlog, period, sig, T




