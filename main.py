from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#my_data = genfromtxt('F:\Programacion\Python\Deep Learning\Competition\DATASET\\train.csv', delimiter=',')

data = pd.read_csv('F:\Programacion\Python\Deep Learning\Competition\DATASET\\train.csv')

my_data= data.to_numpy()

#filter by experiment
exp1 = my_data[my_data[:,1]=="CA"]
exp2 = my_data[my_data[:,1]=="DA"]
exp3 = my_data[my_data[:,1]=="SS"]

#filter by pair of pilot
pair = 1
p1 = exp1[exp1[:,0]== pair]

#filter by seats

seat = 1

p1 = p1[p1[:,3]==seat]

p1 = p1[p1[:,2].argsort()]


points = 50000
numOfplot = 1



for i in range(numOfplot):
    plt.subplot(2,1,1)
    
    plt.plot(p1[0:points-1,2],p1[0:points-1,i+4])
    plt.subplot(2,1,2)
    plt.plot(p1[0:points-1,2],p1[0:points-1,27])

