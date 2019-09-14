
# Libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Data loading
data = pd.read_csv("train.csv")

# Replace event names
data['event'] = data['event'].str.replace('A','1')
data['event'] = data['event'].str.replace('B','2')
data['event'] = data['event'].str.replace('C','3')
data['event'] = data['event'].str.replace('D','4')


# Data slicing for computational purposes
data1 = data[0:900000]

data1 = data


# Parsing data
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

# Example of use
signals["CA_cr2_s1"].head()