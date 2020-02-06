from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])

# print(x_train[0][:][:][:])
print(x_train.shape)
print(x_test.shape)
# print(type(x_train))

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)
# print(y_test.shape)

model = Sequential()
model.add(Dense(320, activation='relu', input_shape=(3072,)))
model.add(BatchNormalization())
model.add(Dense(160, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(80, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(40, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(20, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10 , activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='auto')
hist = model.fit(x_train, y_train, 
                 batch_size=100, 
                 epochs=10, 
                 validation_split=0.2, 
                 callbacks=[earlystopping], 
                 verbose=1)

acc = model.evaluate(x_test, y_test)
print(acc)


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model loss, accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()