from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D , Dense , Flatten , MaxPooling2D , UpSampling2D
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255
# print(x_train.shape)
# print(x_test.shape)
# print(type(x_train))

x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)
# print(x_train.shape)
# print(x_test.shape)

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)
# print(y_test.shape)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(28*28,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10 , activation='softmax'))

# model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='auto')
model.fit(x_train, y_train, batch_size=100, epochs=20, validation_split=0.2, callbacks=[earlystopping], verbose=1)

acc = model.evaluate(x_test, y_test)

print(acc)