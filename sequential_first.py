from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32,input_shape=(784,)),
    # input_shape must be tuple
    # equivalent to Dense(32,input_dim=784)
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True))

model.fit(x_train,y_train,epochs=5,batch_size=32)

# Alternatively, you can feed batches to your model manually:
# model.train_on_batch(x_batch,y_batch)

loss_and_metrics = model.evaluate(x_test,y_test,batch_size=128)
# Or generate predictions on new data:
# classes = model.predict(x_test,batch_size=128)