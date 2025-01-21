# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib.figure import figaspect

import matplotlib as mpl

import numpy as np
from scipy import signal
from matplotlib import rc
from sympy.ntheory import factorint
import os

__author__ = 'Alex'

mpl.rcParams['grid.linewidth'] = 0.3
mpl.rcParams['grid.linestyle'] = '-'
mpl.rcParams['grid.alpha'] = 0.6
mpl.rcParams['grid.color'] = 'gray'

mpl.rcParams['axes.linewidth'] = 0.5

mpl.rcParams['xtick.major.size'] = 1.5
mpl.rcParams['ytick.major.size'] = 1.5

mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2

labelpad = 0


mpl.rcParams['figure.figsize'] = (8.0/2.54, 6.0/2.54) # 6 cm * 8 cm
mpl.rcParams['figure.dpi'] = 300

#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Verdana'


mpl.rcParams['font.size'] = 4

#mpl.rcParams[''] =

mpl.rcParams['figure.subplot.left'] = 0.07
mpl.rcParams['figure.subplot.right'] = 0.995
mpl.rcParams['figure.subplot.bottom'] = 0.08
mpl.rcParams['figure.subplot.top'] = 0.94



path = "./16-08-2019-kran/"
# path = "./"
ASensetivity = (0.31 / 9.8)
BSensetivity = (0.5 / 9.8)

TO_VOLT = 6.1038882E-04

MaxFreq = 400.00
MinFreq = 0

Fdisc = 2.5e4
ADA_BASE_FREQ_C = 3.525e5
ADA_BASE_FREQ = 3.5e5
dt = 1.0 / (ADA_BASE_FREQ_C / ADA_BASE_FREQ * Fdisc)

print dt, 1.0 / dt

NOICE_RATE_ADC = 2
NOICE_RATE_VOLT = NOICE_RATE_ADC * TO_VOLT
NOICE_RATE_ACCEL = NOICE_RATE_ADC * TO_VOLT / max(ASensetivity, BSensetivity)

N_RESAMPLE = 1e4

FACTOR_LIMIT = 128

convertToDisplacement = False

useFiltering = False


def bestFFTlength(n):
    while max(factorint(n)) >= FACTOR_LIMIT and max(factorint(n).values()) < 3:
        n -= 1
    return n


def ride_comfort_func(val):
    x = np.array([0, 1, 2, 4, 8, 16, 32, 410])

    k = np.array([0.29, 0.5, 0.71, 1, 1, 0.5, 0, 0])

    n = x.size
    j = 0
    for i in range(n - 1):
        if x[i] <= val < x[i + 1]:
            j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res

def kb(val):
    x = np.array([0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
                  50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])

    k = np.array([5.20E-02, 5.20E-02, 5.13E-02, 5.14E-02, 5.17E-02, 5.19E-02,
                  5.17E-02, 5.20E-02, 5.22E-02, 5.23E-02, 5.25E-02, 5.27E-02,
                  5.35E-02, 5.45E-02, 5.52E-02, 5.67E-02, 5.88E-02, 6.05E-02,
                  6.26E-02, 4.92E-02, 3.31E-02])

    n = x.size
    j = 0
    for i in range(n - 1):
        if x[i] <= val < x[i + 1]:
            j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res


def ka(val):
    x = np.array([0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
                  50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])
    k = np.array([3.33E-02, 3.33E-02, 3.28E-02, 3.30E-02, 3.29E-02, 3.27E-02,
                  3.28E-02, 3.27E-02, 3.29E-02, 3.31E-02, 3.35E-02, 3.40E-02,
                  3.40E-02, 3.52E-02, 3.67E-02, 3.92E-02, 4.35E-02, 4.58E-02,
                  4.81E-02, 4.27E-02, 4.31E-02])

    n = x.size
    j = 0
    for i in range(n - 1):
        if x[i] <= val < x[i + 1]:
            j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res


def kd(val):
    x = np.array([0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
                  50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 315.0, 400.0])
    k = np.array([2.31E-02, 2.31E-02, 2.11E-02, 2.17E-02, 2.25E-02, 2.14E-02, 2.14E-02, 2.23E-02, 2.28E-02,
                  1.79E-02, 1.83E-02, 1.81E-02, 1.71E-02, 1.89E-02, 1.37E-02, 1.49E-02])

    n = x.size
    j = 0
    for i in range(n - 1):
        if x[i] <= val < x[i + 1]:
            j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res


def ke(val):
    x = np.array([0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
                  50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])
    k = np.array([1.99E-02, 1.99E-02, 2.18E-02, 2.22E-02, 2.19E-02, 2.19E-02, 2.12E-02, 2.12E-02, 2.24E-02,
                  2.17E-02, 1.89E-02, 2.00E-02, 2.02E-02, 2.25E-02, 1.90E-02, 2.21E-02, 2.88E-02, 3.33E-02])

    n = x.size
    j = 0
    for i in range(n - 1):
        if x[i] <= val < x[i + 1]:
            j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res


def kf(val):
    x = np.array([0, 16.0, 20.0, 25.0, 31.0, 40.0,
                  50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])
    k = np.array([2.20E-02, 2.20E-02, 2.21E-02, 2.08E-02, 2.10E-02, 2.12E-02, 2.11E-02,
                  2.13E-02, 2.18E-02, 1.59E-02, 1.73E-02, 1.68E-02, 1.70E-02, 1.70E-02, 1.17E-02, 1.02E-02])

    n = x.size
    j = 0
    for i in range(n - 1):
        if x[i] <= val < x[i + 1]:
            j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res


def kfreq(x):
    a = -0.96864463
    b = 620.93631
    c = -0.0037144418
    d = -0.0008899866

    return (a + b * x) / (1 + x * (c + d * x)) / 60.0


def filter(rho, freq):
    for i in range(rho.size):
        if not (0 < freq[i] < MaxFreq):
            rho[i] = 0
    return rho


def oddArray(x):
    if 0 != (x.size % 2):
        x = np.resize(x, (x.size - 1))
    return x


def process(x, ch):
    k = {0: lambda x: ka(x),
         1: lambda x: kb(x),
         # 2 is used for frequency
         3: lambda x: kd(x),
         4: lambda x: ke(x),
         5: lambda x: kf(x)}[ch]

    print('Doing FFT..')
    x_fft = np.fft.rfft(x)

    rho = np.abs(x_fft)
    phi = np.angle(x_fft)

    rho[0] = 0  # drop constant part

    freq = np.fft.fftfreq(x.size, d=dt)
    fr = freq[0:freq.size / 2 + 1]
    fr[-1] = np.abs(fr[-1])
    print('OK')
    print NOICE_RATE_VOLT
    print rho, rho / rho.size, np.max(rho), np.max(rho / rho.size), fr[1]

    ride_comf_val = 0

    print('Correcting spectrum..')

    for i in range(1, rho.size):
        if fr[i] > MaxFreq or fr[i] < 0.3:
            rho[i] = 0.0
            continue

        if rho[i] / rho.size < NOICE_RATE_VOLT and useFiltering:
            rho[i] = 0.0

        rho[i] = rho[i] / k(fr[i])

        if convertToDisplacement:
            rho[i] /= (2.0 * np.pi * fr[i]) ** 2


    print('OK')
    #    plt.plot(rho)
    #    rho = filter(rho,fr)
    #    plt.plot(rho)
    #    plt.show()
    print('Doing inverse FFT..')

    Amp = np.fft.irfft(rho * np.exp(1.j * phi) * 2.0)

    print('OK')
    return (Amp, rho / rho.size, fr)


def plot(x, y, minn, maxn, filename, col=0):
    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000', '#00FFFF', '#FF00FF']
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
    ax.grid(True, which='both', )
    ax.locator_params(axis='both', tight=True, nbins=10)
    plt.plot(x[minn:maxn], y[minn:maxn], lw=2, color=colors[col])
    plt.savefig(filename, dpi=300)
    #    plt.show()
    plt.close()


def plot2(x, y, minn, maxn, filename, col=0, ride_comf_val = -1):
    colors = ['#0000FF', '#00CC00', '#FFFF00', '#CC0000', '#00CCCC', '#CC00CC']

    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')  # set sci format outside (0.001;1000)
    #ax.grid(True, which='both')

    ax.grid(True,which="majorminor")
    ax.locator_params(axis='both', tight=True, nbins=10)

    y = y - np.mean(y)
    nstart = int(0.1 * y.size)
    nend = nstart + bestFFTlength(int(0.9 * y.size) - nstart)
    print "PSD length:", nend - nstart, " with factorization ", factorint(nend - nstart)
    ny = y[nstart:nend]
    nx = np.arange(0, ny.size * dt, dt)[0:ny.size]

    print "Plotting time series...", nx.size, ny.size
    plt.subplot(211)
    plt.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
    if convertToDisplacement:
        plt.title(u'СКЗ = ' + u"{0:.2e}".format(ny.std()) + u' м')
    else:
        plt.title(u"СКЗ = {0:.2f} м/с²".format(ny.std(), ride_comf_val) )
    plt.plot(nx, ny, lw = 1, color=colors[col])
    plt.xlabel(u't, с', labelpad = labelpad)
    if convertToDisplacement:
        plt.ylabel(u'u, м', labelpad = labelpad)
    else:
        plt.ylabel(u'a, м/с²', labelpad = labelpad)
    plt.grid(True)
    plt.xlim(0, nx[-1])

    if convertToDisplacement:
        plt.ylim(-0.1, 0.1)  # set y scale to displacement 0.1 m which is equivalent to 40 m/s2 at ~3 Hz
    else:
        plt.ylim(-40, 40)  # set y scale to -40:40 m/s2
    print "OK"

    print "Plotting PSD series..."

    ax1 = plt.subplot(212)


    (ff, psd_period) = signal.periodogram(ny, fs = 1.0 / dt, return_onesided=True)

    ax1.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')

    if convertToDisplacement:
        ax1.set_ylabel(u'S, м²/Гц', labelpad = labelpad)
        ax1.set_xlim(0.3, 10)
        ax1.plot(ff, psd_period, lw = 1, color=colors[col])
        ax1.locator_params(axis='x', nbins=20)
    else:
        ax1.set_ylabel(u'S, (м/с²)²/Гц', labelpad = labelpad)
        #max_subplot_freq = 10
        #axins = inset_axes(ax1, loc=1, width="40%", height="40%")
        #mark_inset(ax1,axins,loc1=2, loc2=4, fc="none", ec="0.5")
        #axins.set_xlim(0.3, max_subplot_freq)
        #axins.locator_params(axis='x', nbins= 10)
        #axins.locator_params(axis='y', nbins= 4)

        #axins.plot(ff[ff < max_subplot_freq], psd_period[ff < max_subplot_freq], lw = 1, color=colors[col])
        #axins.grid(True)
        #axins.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')

        ax1.set_xlim(0.3, 400)
        ax1.plot(ff, psd_period, lw = 1, color=colors[col])


    ax1.set_xlabel(u'f, Гц', labelpad = labelpad)

    ax1.grid(True, which='both')

    print "OK"
    print "Saving png..."
    plt.savefig(filename, dpi=300)
    #plt.show()
    plt.close()
    print "OK"


def preproc(res):
    x1 = np.zeros(res[:, 3].size)
    x2 = np.zeros(res[:, 3].size)
    m_freq = np.zeros(res[:, 3].size)
    x3 = np.zeros(res[:, 3].size)
    x4 = np.zeros(res[:, 3].size)
    x5 = np.zeros(res[:, 3].size)
    i = 0
    k = 0
    res_size = res[:, 3].size
    while i < res_size - 1:
        if res[i, 3] > res[i + 1, 3]:
            j = i + 1
            while res[i, 3] > res[j, 3]:
                j += 1
        print "Stripping part from " + str(i) + " to " + str(j)
        i = j

        x1[k] = res[i, 0]
        x2[k] = res[i, 1]
        m_freq[k] = res[i, 2]
        x3[k] = res[i, 3]
        x4[k] = res[i, 4]
        x5[k] = res[i, 5]

        k += 1
        i += 1

    x1 = np.trim_zeros(x1)
    x2 = np.trim_zeros(x2)
    m_freq = np.trim_zeros(m_freq)
    x3 = np.trim_zeros(x3)
    x4 = np.trim_zeros(x4)
    x5 = np.trim_zeros(x5)
    return (x1, x2, m_freq, x3, x4, x5)


def loadfile(filename):
    print filename
    res = np.loadtxt(path + filename, dtype='float', comments='#', delimiter='\t')

    #    (x1,x2,m_freq,x3,x4,x5) = preproc(res)
    num_of_signals = res[0, :].size
    num_of_counts = int(res[:, 0].size)

    old_num = num_of_counts
    num_of_counts = bestFFTlength(num_of_counts)

    print "Initial array length is ", old_num
    if old_num != num_of_counts:
        print "Trancating it length to ", num_of_counts
        print old_num - num_of_counts, " elements have been deleted"

    print "num_of_counts factorization is:", factorint(num_of_counts)

    #    print num_of_signals, num_of_counts
    t = np.arange(0, num_of_counts * dt, dt)[0:num_of_counts]

    res_new = np.zeros((num_of_counts, num_of_signals))
    res_new[:, -1] = t
    for i in range(num_of_signals - 1):

        x = res[0:num_of_counts, i] * TO_VOLT

        #if (i in [0,2,3,4]):
        #    continue

        if i == 2:
            continue
            amp = kfreq(x)
            minn = 0
            maxn = x.size
            freq = rho = np.array([])

        else:
            (amp, rho, freq) = process(x, i)
            maxn = int(300.0 / freq[1])
            minn = int(MinFreq / freq[1])

        res_new[0:amp.size, i] = amp[0:amp.size]

        ride_comf_val = 0
        for j in range(rho.size):
            ride_comf_val += (ride_comfort_func(freq[j])*rho[j])**2.0

        ride_comf_val = np.sqrt(ride_comf_val)


        print 'ride_comf_value = ',ride_comf_val
        # if (num_of_counts > N_RESAMPLE):
        #    amp = signal.resample(amp, N_RESAMPLE)
        #    t = np.arange(0, t[-1], t[-1]/N_RESAMPLE)[0:amp.size]
        #    #freq = signal.resample(freq, N_RESAMPLE)
        #    print "resampling is required"

        print "Plotting " + str(i) + " channel"

        # plt.plot(rho, freq)
        # psd = plt.psd(amp, NFFT=amp.size, Fs = 1.0/dt)
        # plt.close()

        #       plt.plot(psd[1],psd[0])
        #        plt.plot(freq, rho)
        #        plt.plot(freq, corr)
        #        plt.show()
        plot2(t, amp, minn, maxn, path + "out/" + filename + "--signal+psd--" + str(i) + "---.png", i, ride_comf_val)
        #        exit(1)
        #        (ff,psd_period) = signal.periodogram(amp,fs = 1.0/dt, return_onesided=True)


        # plot(ff, psd_period, minn, maxn, path + "out/"+ filename + "---"+str(i)+"---spectrum2_"+str(MinFreq)+"-"+str(MaxFreq)+".png", i)


        # if (rho.size):
        #    plot(freq, rho, minn, maxn, path + "out/"+ filename + "---"+str(i)+"---spectrum_"+str(MinFreq)+"-"+str(MaxFreq)+".png", i)

        # if (rho.size):
        #    plot(psd[1], psd[0], minn, maxn, path + "out/"+ filename + "---"+str(i)+"---psd_"+str(MinFreq)+"-"+str(MaxFreq)+".png", i)

        # plot(t, amp, 0, t.size, path + "out/"+ filename +"---"+str(i)+"---.png", i )
        # plot(t, amp, 0, 0.2/t[1], path + "out/"+ filename +"---"+str(i)+"---0.2s.png")

        #        plt.show()
        #       plt.close()
        # exit(0)
        print "OK"

    print "Saving..."
    #np.savetxt(path + "out/" + filename + "_out.txt", [0], delimiter='\t')
    np.savetxt(path +"out/" + filename+"_out.txt", res_new, delimiter='\t')
    print "OK"

    return


def main():

    if (False):
        freq = np.linspace(0, 50, 100)
        ride_coef = np.zeros_like(freq)
        for i in range(freq.size):
            ride_coef[i] = ride_comfort_func(freq[i])

        plt.plot(freq,ride_coef)
        plt.xlabel(u'f, Гц',labelpad=labelpad)
        plt.ylabel(u'k(f)',labelpad=labelpad)
        plt.grid(True)
        plt.savefig('1.png',dpi=300)
        plt.show()

        plt.close()

    #    loadfile("class_1000_datch1_datch2_mbax")
    for filename in os.listdir(path):
        if os.path.isfile(path + filename):
            if not os.path.isfile(path + "out/" + filename + "_out.txt"):
                loadfile(filename)
            else:
                print "Skipping file " + filename


main()
