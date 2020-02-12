import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

num_classes=1
batch_size=1
epochs=50
num_data=1313

data = pd.read_csv('bd_train.csv')
data.loc[lambda data: data['Class']>0.5,'Class']=1
data.loc[lambda data: data['Class']<0.5,'Class']=0
data_a=data.iloc[:,2:].values
# data_norm=preprocessing.MinMaxScaler().fit_transform(data_a)
data_norm=data_a


x_train=data_norm[:,:num_data]
_y_train=data['Class'].values
y_train=_y_train.reshape(len(_y_train),1)

data2 = pd.read_csv('bd_test.csv')
data2.loc[lambda data2: data2['Class']>0.5,'Class']=1
data2.loc[lambda data2: data2['Class']<0.5,'Class']=0
data2_a=data2.iloc[:,2:].values
# data2_norm=preprocessing.StandardScaler().fit_transform(data2_a)
data2_norm=data2_a

x_test=data2_norm[:,:num_data]
_y_test=data2['Class'].values
y_test=_y_test.reshape(len(_y_test),1)



model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(num_data,)))#1313
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict(x_test)
y_pred = (predictions > 0.5)

print(confusion_matrix(y_test,y_pred))

# 5. 학습효과분석
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'], loc=0)

def plot_acc(history):
    plt.plot(history.history['accuracy']) # 오류나면 'accuracy' -> 'acc'
    plt.plot(history.history['val_accuracy']) # 오류나면 'val_accuracy' -> 'val_acc'
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'], loc=0)

#def plot_loss_acc(history, ax=None, **kwargs):
#    ax = 

"""
plot_loss(history)
plt.show()
plot_acc(history)
plt.show()
"""

# plot_loss, plot_acc 함수 안쓰고 그림
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-')
plt.plot(history.history['val_loss'], 'r:')
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel('Epoch')
plt.legend(['Train','Test'], loc=0)
plt.subplot(1, 2, 2)
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.plot(history.history['accuracy'], 'b-')
plt.plot(history.history['val_accuracy'], 'r:')
plt.legend(['Train','Test'], loc=0)
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()