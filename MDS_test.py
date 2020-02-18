import pandas as pd
import numpy as np
from sklearn.manifold import MDS


num_classes=1
batch_size=1
epochs=20
num_data=100
#3874

save_folder_path = "C:/ML_2020/2.sweetness/Datas/"
data = pd.read_csv(save_folder_path + 'sweetness_MD_already.csv')
_data=data.iloc[:,1:].values
data_a=_data[:,1:]
# data_norm=preprocessing.MinMaxScaler().fit_transform(data_a)
data_norm=data_a

x_transformed = MDS(n_components=2).fit_transform(data_norm)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x_transformed[:,0],x_transformed[:,1])
plt.show()