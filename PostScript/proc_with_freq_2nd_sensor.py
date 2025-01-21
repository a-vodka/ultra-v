__author__ = 'Alex'

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, misc

import os

path = "./pivo_13_11_2014/hummer_test/"
#path = "./"
ASensetivity = (0.31/9.8)
BSensetivity = (0.5/9.8)

TO_VOLT = 6.1038882E-04

MaxFreq = 400.00
MinFreq = 0

Fdisc = 2.5e4
ADA_BASE_FREQ_C = 3.525e5
ADA_BASE_FREQ   = 3.5e5
dt = 1.0/(ADA_BASE_FREQ_C/ADA_BASE_FREQ*Fdisc)

print dt, 1.0/dt

NOICE_RATE_ADC 	=	4
NOICE_RATE_VOLT =	NOICE_RATE_ADC * TO_VOLT
NOICE_RATE_ACCEL=	NOICE_RATE_ADC * TO_VOLT / max(ASensetivity,BSensetivity)

N_RESAMPLE = 1e4

def kb(val):

    x = np.array ([0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])

    k = np.array ([5.20E-02, 5.20E-02, 5.13E-02, 5.14E-02, 5.17E-02, 5.19E-02,
					5.17E-02, 5.20E-02, 5.22E-02, 5.23E-02, 5.25E-02, 5.27E-02,
					5.35E-02, 5.45E-02, 5.52E-02, 5.67E-02, 5.88E-02, 6.05E-02,
					6.26E-02, 4.92E-02, 3.31E-02] )

    n = x.size
    j = 0
    for i in range(n-1):
        if x[i] <= val < x[i + 1]:
            j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res



def ka(val):
    x = np.array([0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])
    k = np.array([3.33E-02, 3.33E-02, 3.28E-02, 3.30E-02, 3.29E-02, 3.27E-02,
				3.28E-02, 3.27E-02, 3.29E-02,3.31E-02, 3.35E-02, 3.40E-02,
				3.40E-02, 3.52E-02, 3.67E-02, 3.92E-02, 4.35E-02, 4.58E-02,
				4.81E-02, 4.27E-02, 4.31E-02 ])

    n = x.size
    j = 0
    for i in range(n-1):
        if x[i] <= val < x[i + 1]:
           j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res

def kd(val):
    x = np.array([0,10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 315.0, 400.0])
    k = np.array([2.31E-02, 2.31E-02, 2.11E-02, 2.17E-02, 2.25E-02, 2.14E-02, 2.14E-02, 2.23E-02, 2.28E-02,
	1.79E-02, 1.83E-02, 1.81E-02, 1.71E-02, 1.89E-02, 1.37E-02, 1.49E-02 ])

    n = x.size
    j = 0
    for i in range(n-1):
        if x[i] <= val < x[i + 1]:
           j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res
def ke(val):
    x = np.array([0,10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,  250.0, 315.0, 400.0])
    k = np.array([1.99E-02, 1.99E-02, 2.18E-02, 2.22E-02, 2.19E-02, 2.19E-02, 2.12E-02, 2.12E-02, 2.24E-02,
2.17E-02, 1.89E-02, 2.00E-02, 2.02E-02, 2.25E-02, 1.90E-02, 2.21E-02, 2.88E-02, 3.33E-02 ])

    n = x.size
    j = 0
    for i in range(n-1):
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
    for i in range(n-1):
        if x[i] <= val < x[i + 1]:
           j = i

    res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j])
    return res

def kfreq(x):
	a =	-0.96864463
	b =	620.93631
	c =	-0.0037144418
	d =	-0.0008899866

	return (a+b*x)/(1+x*(c+d*x))/60.0


def filter(rho, freq):
    for i in range(rho.size):
        if not (0 < freq[i] < MaxFreq):
            rho[i]=0
    return rho

def oddArray(x):
    if 0 != (x.size % 2):
        x = np.resize(x,(x.size-1))
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

    rho[0] = 0 # drop constant part

    freq = np.fft.fftfreq(x.size, d=dt)
    fr = freq[0:freq.size/2+1]
    fr[-1] = np.abs(fr[-1])
    print('OK')

    print('Correcting spectrum..')
    for i in range(1,rho.size):
        if fr[i]>MaxFreq:
            rho[i]=0.0
            continue

        rho[i] = rho[i]/k(fr[i])

#        if rho[i]/rho.size < NOICE_RATE_ACCEL:
#            rho[i]=0.0
            
    print('OK')

    rho = filter(rho,fr)
    print('Doing inverse FFT..')

    Amp = np.fft.irfft(rho*np.exp(1.j*phi)*2.0)
    print('OK')
    return (Amp, rho/rho.size, fr)

def plot(x, y, minn, maxn, filename):
    ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax.grid(True, which='both',)
    ax.locator_params(axis='both',tight=True, nbins=10)
    plt.plot(x[minn:maxn],y[minn:maxn],lw=2)
    plt.savefig(filename,dpi=300)
#    plt.show()
    plt.close()

def preproc(res):
        x1 = np.zeros(res[:,3].size)
        x2 = np.zeros(res[:,3].size)
        m_freq = np.zeros(res[:,3].size)
        x3 = np.zeros(res[:,3].size)
        x4 = np.zeros(res[:,3].size)
        x5 = np.zeros(res[:,3].size)
        i = 0
        k = 0
        res_size = res[:,3].size
        while (i < res_size - 1):
            if (res[i,3] > res[i+1,3]):
                j = i + 1
                while (res[i,3] > res[j,3]):
                    j += 1
            print "Stripping part from " + str(i) + " to " + str(j)
            i = j

            x1[k] = res[i,0]
            x2[k] = res[i,1]
            m_freq[k] = res[i, 2]
            x3[k] = res[i,3]
            x4[k] = res[i,4]
            x5[k] = res[i,5]

            k += 1
            i += 1

        x1 = np.trim_zeros(x1)
        x2 = np.trim_zeros(x2)
        m_freq = np.trim_zeros(m_freq)
        x3 = np.trim_zeros(x3)
        x4 = np.trim_zeros(x4)
        x5 = np.trim_zeros(x5)
        return (x1,x2,m_freq,x3,x4,x5)

def loadfile(filename):
    print filename
    res = np.loadtxt(path+filename, dtype='float', comments='#', delimiter='\t')
    
#    (x1,x2,m_freq,x3,x4,x5) = preproc(res)
    num_of_signals = res[0,:].size
    num_of_counts = res[:,0].size
    print num_of_signals, num_of_counts
    t = np.arange(0, num_of_counts*dt, dt)[0:num_of_counts]

    res_new = np.zeros((num_of_counts, num_of_signals))
    res_new[:,-1] = t
    for i in range(num_of_signals-1):

        x = res[:, i]*TO_VOLT

        if (i == 2):
            amp = kfreq(x)
            minn = 0
            maxn = x.size
            freq = rho = np.array([])
            
        else:
            (amp, rho, freq) = process(x, i)
            maxn = int(MaxFreq/freq[1])
            minn = int(MinFreq/freq[1])

        res_new[0:amp.size,i] = amp[0:amp.size]

        if (num_of_counts > N_RESAMPLE):
            amp = signal.resample(amp, N_RESAMPLE)
            t = np.arange(0, t[-1], t[-1]/N_RESAMPLE)[0:amp.size]
            #freq = signal.resample(freq, N_RESAMPLE)
            print "resampling is required"

        print "Plotting "+str(i)+" channel"

        if (rho.size):
            plot(freq, rho, minn, maxn, path + "out/"+ filename + "---"+str(i)+"---spectrum_"+str(MinFreq)+"-"+str(MaxFreq)+".png")

        plot(t, amp, 0, t.size, path + "out/"+ filename +"---"+str(i)+"---.png")
        #plot(t, amp, 0, 0.2/t[1], path + "out/"+ filename +"---"+str(i)+"---0.2s.png")

        print "OK"

    print "Saving..."
    np.savetxt(path +"out/" + filename+"_out.txt", res_new, delimiter='\t')
    print "OK"

    return

def main():
#    loadfile("class_1000_datch1_datch2_mbax")
    for filename in os.listdir(path):
        if (os.path.isfile(path+filename)):
            if (not os.path.isfile(path +"out/" + filename+"_out.txt")):
                loadfile(filename)
            else:
                print "Skipping file "+filename
            



main()






