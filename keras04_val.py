import numpy as np
x_train=np.array([1,2,3,4,5,6,7,8,9,10])
y_train=np.array([1,2,3,4,5,6,7,8,9,10])
x_test=np.array([11,12,13,14,15,16,17,18,19,20])
y_test=np.array([11,12,13,14,15,16,17,18,19,20])





from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(5, input_shape=(1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

model.fit(x_train, y_train, epochs=20, batch_size=32)

eval=model.evaluate(x_test, y_test,  steps=20)
print(eval[1])