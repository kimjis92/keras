from numpy import array
from keras.models import Sequential
from keras.layers import Dense

x = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12],
           [20000, 30000, 40000], [30000, 40000, 50000], [40000, 50000, 60000], [100, 200, 300]])
y = array([4, 5, 6, 7, 8, 9 ,10, 11, 12, 13, 50000, 60000, 70000, 400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler



x_pre=array([300, 260, 270])

scaler2 = StandardScaler()
scaler2.fit(x)
x = scaler2.transform(x)
x_pre=x_pre.reshape(1,-1)
x_pre = scaler2.transform(x_pre)



scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_pre = scaler.transform(x_pre)





x_train=x[:11]
x_test=x[11:]


y_train=y[:11]
y_test=y[11:]

model=Sequential()

model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, batch_size=1, epochs=100)


aaa=model.evaluate(x_test, y_test, batch_size=1)

print(aaa)
print(x_pre.shape)



x_pre=x_pre.reshape(1,3)

y_pre=model.predict(x_pre)

print('y_pre : ', y_pre)
print('x_pre : ', x_pre)


from sklearn.metrics import r2_score
y_predict=model.predict(x_test, batch_size=1)
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

