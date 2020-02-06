from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D , Dense , Flatten , MaxPooling2D , UpSampling2D , BatchNormalization
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
# print(x_train[0][:][:][:])
# print(x_train.shape)
# print(x_test.shape)
# print(type(x_train))

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)
# print(y_test.shape)

model = Sequential()
model.add(Conv2D(320, (2,2), padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Conv2D(160,(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(80,(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(40,(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(20,(2,2),padding='same'))
model.add(BatchNormalization())
model.add(UpSampling2D(2))
model.add(Flatten())
model.add(Dense(10 , activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlystopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='auto')
hist = model.fit(x_train, y_train, 
                 batch_size=10, 
                 epochs=100, 
                 validation_split=0.2, 
                 callbacks=[earlystopping], 
                 verbose=1)

acc = model.evaluate(x_test, y_test)
print(acc)


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model loss, accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()