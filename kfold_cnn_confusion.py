import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score,cross_val_predict

num_classes=1
batch_size=1
epochs=10
num_data=1296
#1313
data = pd.read_csv('bd_train.csv')
data.loc[lambda data: data['Class']>0.5,'Class']=1
data.loc[lambda data: data['Class']<0.5,'Class']=0
data_a=data.iloc[:,2:num_data+2].values
# data_norm=preprocessing.MinMaxScaler().fit_transform(data_a)
data_norm=data_a.reshape(data_a.shape[0],36,36,1)


x_train=data_norm
_y_train=data['Class'].values
y_train=_y_train.reshape(len(_y_train),1)

data2 = pd.read_csv('bd_test.csv')
data2.loc[lambda data2: data2['Class']>0.5,'Class']=1
data2.loc[lambda data2: data2['Class']<0.5,'Class']=0
data2_a=data2.iloc[:,2:num_data+2].values
# data2_norm=preprocessing.StandardScaler().fit_transform(data2_a)
data2_norm=data2_a.reshape(data2_a.shape[0],36,36,1)

x_test=data2_norm
_y_test=data2['Class'].values
y_test=_y_test.reshape(len(_y_test),1)

X=np.concatenate((x_test,x_train))
Y=np.concatenate((y_test,y_train))

def create_model():
    model = Sequential()
    model.add(Conv2D(5,(2,2),input_shape=(36,36,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(5,(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

#random seed
seed=7
np.random.seed(seed)

#sklearn model
sk_model = KerasClassifier(build_fn=create_model,epochs=epochs,batch_size=batch_size,verbose=0)#need epoch, batch_size, verbose?

kfold=KFold(n_splits=5,shuffle=True,random_state=seed)
# result=cross_val_score(sk_model,X,Y,cv=kfold)
# print(result)
Y_pred=cross_val_predict(sk_model,X,Y,cv=kfold)
print(confusion_matrix(Y,Y_pred))
