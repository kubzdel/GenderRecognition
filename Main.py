import wave, struct

import parabolic as parabolic
from numpy.fft import fft
from scipy.io.wavfile import read
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import wave
import numpy
from numpy.fft import rfft
from scipy.signal import blackmanharris, kaiser, decimate, resample
import glob

def processFile(filename):
    fp = wave.open(filename)
    N = fp.getnframes()
    channels = fp.getnchannels()
    dstr = fp.readframes(N*channels)
    data = numpy.fromstring(dstr, numpy.int16)
    data = numpy.reshape(data, (-1,channels))
    data2= data[:,channels-1]
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(121)
    ax.plot(data2)
    blackman = data2*kaiser(len(data2),14)
    f=rfft(blackman)
    i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
    print (i)
    f2=f[50:1200]
    ax = fig.add_subplot(222)
    ax.plot(f2)
#plt.show()

# success = 0
# files = glob.glob('train\*.wav')
# for file in files:
#     sex = processFile(file)
#     if sex == file[5]:
#         success+=1
# accuraccy = success/len(files)
# print(accuraccy)


fp = wave.open('train/001_K.wav')
N = fp.getnframes()
channels = fp.getnchannels()
frate = fp.getframerate()
print(frate)
T= frate*N
df=1/T
dstr = fp.readframes(N*channels)
data = numpy.fromstring(dstr, numpy.int16)
data = numpy.reshape(data, (-1,channels))
data2= data[:,channels-1]
fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(221)
ax.plot(data2)
data3= resample(data2,250)
# data3= decimate(data2,50)
# data3= decimate(data2,50)
# data3= decimate(data2,50)
# data3= decimate(data2,50)
ax = fig.add_subplot(222)
ax.plot(abs(data3))
f=rfft(data3)
ax = fig.add_subplot(223)
ax.plot(f)
plt.show()
for i in f:
    if i <50 or i>250:
        i = 0
idx = np.argmax(np.abs(f))
freq2 = 250* idx/len(f)
print(freq2)




# blackman = data2*blackmanharris(len(data2))
# f=rfft(blackman)
# f = abs(f)
# freqs =np.array([df*n if n<N/2 else df*(n-N) for n in range(N)])
# for i in range(0,len(freqs)-1):
#     print(freqs[i])
# max = max(freqs)
# print(max)
# print(freqs.max())
# idx = np.argmax(np.abs(f))
# freq = freqs[idx]
# freq2 = frate* idx/len(f)
# freq_in_hertz = abs(freq * frate)
# print(freq_in_hertz)
# print(freq2)
# i = np.argmax(abs(f))  # Just use this for less-accurate, naive version
# f2=f[0:1200]
# ax = fig.add_subplot(222)
# ax.plot(f2)
# plt.show()