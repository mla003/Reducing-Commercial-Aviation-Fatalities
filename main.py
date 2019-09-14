from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
from scipy.signal import convolve as sig_convolve, fftconvolve, lfilter, firwin
from scipy.ndimage import convolve1d

#my_data = genfromtxt('F:\Programacion\Python\Deep Learning\Competition\DATASET\\train.csv', delimiter=',')

data = pd.read_csv('F:\Programacion\Python\Deep Learning\Competition\DATASET\\train.csv')

my_data= data.to_numpy()

low = scipy.io.loadmat('lowpass.mat')
filt = low['Num']

#RUN FROM HERE

#filter by experiment
state = "SS" # "CA","DA", "SS"

exp = my_data[my_data[:,1]==state]


#filter by pair of pilot
pair = 1
p1 = exp[exp[:,0]== pair]

#filter by seats

seat = 1

p1 = p1[p1[:,3]==seat]

p1 = p1[p1[:,2].argsort()]

start = 1024*18
points = 1024*4
numOfplot = 1
freq = np.array([x for x in range(int(points/2)-1)])*256/points


i = 17
    
signal = np.array(p1[start:start+points-1,i+4],dtype=float)
window = 1#np.hamming(points)
spec = 20*np.log10(abs(np.multiply(np.fft.fft(signal/1000,points),window))[0:int(points/2)-1])   

plt.figure()

plt.subplot(3,1,1)
plt.plot(freq,spec)
plt.ylabel("Magnitude [dB]")
plt.xlabel("Frequency [Hz]")
axes=plt.gca()


plt.subplot(3,1,2)
plt.plot(p1[start:start+points-1,2],signal)

#plt.subplot(4,1,3)
#ss = np.convolve(signal,filt,mode='valid')
#plt.plot(p1[start:start+points-1,2],ss)

plt.subplot(3,1,3)
plt.plot(p1[start:start+points-1,2],p1[start:start+points-1,27])

