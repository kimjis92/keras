from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5] ,[4, 5, 6], [5, 6, 7]])
y = array([4,5,6,7,8])

x = x.reshape(x.shape[0], x.shape[1], 1)  #lstm에 넣기위해 x의 구조를 바꿈

model=Sequential()
model.add(LSTM(8, activation='relu', input_shape=(3,1)))
model.add(Dense(4))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
model.fit(x, y, epochs=100, batch_size=1)


loss, mse, acc=model.evaluate(x, y, batch_size=1)


print('loss : ', loss)
print('mse : ', mse)
print('acc : ', acc)

x_input=array([6, 7, 8])
x_input=x_input.reshape(1, 3, 1)

y_predict=model.predict(x_input)
print(y_predict)