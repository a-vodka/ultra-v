__author__ = 'Alex'

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, misc

import os

path = "./Poz_mash_v_klasse_disbalance2/"
#path = "./"
ASensetivity = (0.31/9.8)
BSensetivity = (0.5/9.8)

TO_VOLT = 6.1038882E-04

MaxFreq = 400.00
MinFreq = 0

Fdisc = 2.5e4
ADA_BASE_FREQ_C = 3.580e5
ADA_BASE_FREQ   = 3.5e5
dt = 1.0/(ADA_BASE_FREQ_C/ADA_BASE_FREQ*Fdisc)

print dt, 1.0/dt

NOICE_RATE_ADC 	=	4
NOICE_RATE_VOLT =	NOICE_RATE_ADC * TO_VOLT
NOICE_RATE_ACCEL=	NOICE_RATE_ADC * TO_VOLT / max(ASensetivity,BSensetivity)

N_RESAMPLE = 0.1e5

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

def kfreq(x):
	a =	-0.96864463
	b =	620.93631
	c =	-0.0037144418
	d =	-0.0008899866

	return (a+b*x)/(1+x*(c+d*x))


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

    if ch == 1:
        k = lambda x: ka(x)
    else:
        k = lambda x: kb(x)

    print('Do FFT..')
    x_fft = np.fft.rfft(x)

    rho = np.abs(x_fft)
    phi = np.angle(x_fft)

    rho[0] = 0 # drop constant part

    freq = np.fft.fftfreq(x.size, d=dt)
    fr = freq[0:freq.size/2+1]
    fr[-1] = np.abs(fr[-1])
    print('OK')

    print('Correct spectrum..')
    for i in range(1,rho.size):
        if fr[i]>MaxFreq:
            rho[i]=0.0
            continue

        rho[i] = rho[i]/k(fr[i])

#        if rho[i]/rho.size < NOICE_RATE_ACCEL:
#            rho[i]=0.0
            
    print('OK')

    rho = filter(rho,fr)
    print('Do inverse FFT..')

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

            k += 1
            i += 1

        x1 = np.trim_zeros(x1)
        x2 = np.trim_zeros(x2)
        m_freq = np.trim_zeros(m_freq)
        return (x1,x2,m_freq)

def loadfile(filename):
    print filename
    res = np.loadtxt(path+filename, dtype='float', comments='#', delimiter='\t')
    
    (x1,x2,m_freq) = preproc(res)

    x1 = oddArray(x1*TO_VOLT)
    x2 = oddArray(x2*TO_VOLT)
    m_freq = kfreq(oddArray(m_freq*TO_VOLT))/60.0;
    
    m_freq_resampled = signal.resample(m_freq, N_RESAMPLE)
    
    t = np.arange(0, x1.size*dt, dt)[0:x1.size]
    res = np.zeros((t.size,4))

    res[:,3] = t
    res[:,2] = m_freq[0:t.size]
    (amp, rho, freq) = process(x1,1)

    maxn = int(MaxFreq/freq[1])
    minn = int(MinFreq/freq[1])

    res[:,0] = amp[0:t.size]

    amp = signal.resample(amp, N_RESAMPLE)
    t_res = np.arange(0, t[-1], t[-1]/N_RESAMPLE)[0:amp.size]
    
    print "Plotting 1st channel"
    
    plot(freq, rho, minn, maxn, path + "out/"+ filename + "---A---spectrum_"+str(MinFreq)+"-"+str(MaxFreq)+".png")
    plot(t_res, amp, 0, t_res.size, path + "out/"+ filename +"---A---.png")
    #plot(t_res, amp, 0, 0.2/t_res[1], path + "out/"+ filename +"---A---0.2s.png")
    
    print "OK"
    (amp, rho, freq) = process(x2,2)
    res[:,1] = amp[0:t.size]
    amp = signal.resample(amp, N_RESAMPLE)
    print "Plotting 2nd channel"

    plot(freq, rho, minn, maxn, path + "out/"+ filename + "---B---spectrum_"+str(MinFreq)+"-"+str(MaxFreq)+".png")
    plot(t_res, amp, 0, t_res.size, path + "out/"+ filename +"---B---.png")
    #plot(t_res, amp, 0, 0.2/t_res[1], path + "out/"+ filename +"---B---0.2s.png")

    plot(t_res, m_freq_resampled, 0, t_res.size, path + "out/"+ filename + "---freq---.png")
	
    print "OK"
    print "Saving..."
    np.savetxt(path +"out/" + filename+"_out.txt", res, delimiter='\t')
    print "OK"
    return

def main():
#    loadfile("test131Hz")
    for filename in os.listdir(path):
        if (os.path.isfile(path+filename)):
            loadfile(filename)

            



main()






