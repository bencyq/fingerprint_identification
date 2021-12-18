import numpy as np
import pandas as pd
import os
import random

path = 'FVC2002_DB1_A'
file_name = os.listdir(path)
dataset_train = []
dataset_test = []
for i in range(len(file_name)):
    for j in range(i + 1, len(file_name)):
        label = 1 if file_name[i][:-6] == file_name[j][: -6] else 0
        dataset_train.append([file_name[i], file_name[j], label]) if random.random() >= 0.2 else dataset_test.append(
            [file_name[i], file_name[j], label])
df1, df2 = pd.DataFrame(np.array(dataset_train)), pd.DataFrame(np.array(dataset_test))
df1.to_csv('FVC2002_DB1_A.csv', index=False, columns=None)
df2.to_csv('FVC2002_DB1_C.csv', index=False, columns=None)
df = pd.read_csv('FVC2002_DB1_A.csv', index_col=None)
print(df)