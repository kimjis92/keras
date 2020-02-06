from keras.datasets import mnist
from keras.models import Sequential , Model
from keras.layers import Conv2D , Dense , Flatten , MaxPooling2D , LSTM , Input , ConvLSTM2D, Reshape
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

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)
# print(y_test.shape)

""" model = Sequential()
model.add(Conv2D(10, (2,2), input_shape=(28,28,1)))
model.add(Reshape((27*27,10)))
model.add(LSTM(32) )
model.add(Dense(10,activation='softmax')) """
#model.summary()

input1 = Input(shape=(28,28,1))
output1 = Conv2D(40, (2,2), padding='same')(input1)
output1 = Conv2D(20, (2,2), padding='same')(output1)
output1 = Conv2D(10, (2,2), padding='same')(output1)
output1 = Reshape((28*28,10))(output1)
output1 = LSTM(40)(output1)
output1 = Dense(20)(output1)
output1 = Dense(10, activation='softmax')(output1)
model = Model(inputs=input1, outputs=output1)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping = EarlyStopping(monitor='val_accuracy', patience=100, mode='auto')
model.fit(x_train, y_train, batch_size=100, epochs=100, validation_split=0.2, callbacks=[earlystopping], verbose=1)

acc = model.evaluate(x_test, y_test)

print(acc)

