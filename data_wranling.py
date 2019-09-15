
# Libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra


# Data loading
data = pd.read_csv("train.csv")

# Replace event names
data['event'] = data['event'].str.replace('A','0001')
data['event'] = data['event'].str.replace('B','0010')
data['event'] = data['event'].str.replace('C','0100')
data['event'] = data['event'].str.replace('D','1000')


# Data slicing for computational purposes
data1 = data[0:1000]
#data1 = data


# Normalization process

# Array with signals
eeg = ['eeg_fp1', 'eeg_f7', 'eeg_f8',
       'eeg_t4', 'eeg_t6', 'eeg_t5', 'eeg_t3', 'eeg_fp2', 'eeg_o1', 'eeg_p3',
       'eeg_pz', 'eeg_f3', 'eeg_fz', 'eeg_f4', 'eeg_c4', 'eeg_p4', 'eeg_poz',
       'eeg_c3', 'eeg_cz', 'eeg_o2', 'ecg', 'r', 'gsr']


i = 0
# Arrays for means and variances


# Mean and variance calculations

hola = data1.to_numpy()
hola = hola[:,4:27]
hola = np.array(hola, dtype = float)

mean_dat = np.mean(hola,axis =0)
var_dat = np.var(hola,axis =0)

#Data frame creation    
data_stats = {'Signal':eeg, 'Mean':mean_dat, "Variance":var_dat} 
data_stats = pd.DataFrame(data_stats) 
data_stats.set_index("Signal")

# Normalization: Substract mean and divide by variance
#for ind in eeg:  
 #   data1[ind] = data1[ind].apply(lambda x: (x- data_stats[data_stats["Signal"]== ind]["Mean"])/data_stats[data_stats["Signal"]== ind]["Variance"])



tmp = np.divide((hola - mean_dat),var_dat)

# HASTA ACÁ LLEGA. FALTA INCORPORAR AL DATA FRAME

# Parsing data process 
    
    
# We define the vectors contaning each feature
exp = ["CA","DA","SS"]
crew = [1,2,3,4,5,6,7,8,9]
seat = [0,1]

# We define a dictionary which will contain the dataframes

signals = {}

# Parsing according to each feature
for ind1 in exp:
    for ind2 in crew:
        for ind3 in seat:
            name = ind1 + "_cr" + str(ind2) + "_s" + str(ind3)
            temp = data1[data1["experiment"]== ind1]
            temp1 = temp[temp["crew"]== ind2]
            signals[name] = temp1[temp1["seat"]== ind3]
            signals[name].sort_values('time', inplace=True)

# Example of use
print(signals["CA_cr1_s0"].head())
