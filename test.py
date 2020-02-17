import pandas as pd
import numpy as np


save_folder_path = "C:/ML_2020/2.sweetness/Datas/"
data = pd.read_csv(save_folder_path + 'sweetness_MD_already.csv')
_data=data.iloc[:,1:].values
np.random.shuffle(_data)
data_a=_data[:,1:]
# data_norm=preprocessing.MinMaxScaler().fit_transform(data_a)
data_norm=data_a

_y=_data[:,:1]
_y=np.where(_y=='Sweet',1,_y)
_y=np.where(_y=='Non-sweet',0,_y)

n=_y.shape[0]
n=int(n*0.7)
print(n)