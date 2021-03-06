import numpy as np
from sklearn.model_selection import train_test_split
x=np.array(range(1, 101))
y=np.array(range(1, 101))

x_tv, x_test, y_tv, y_test=train_test_split(x, y, test_size=0.2, shuffle=False)
x_train, x_validation, y_train, y_validataion=train_test_split(x_tv, y_tv, test_size=0.25, shuffle=False)



print("x_train", x_train)
print("x_validation", x_validation)
print("x_test", x_test)




from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(5, input_shape=(1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=20, batch_size=1)

eval=model.evaluate(x_test, y_test,  steps=20)

y_predict=model.predict(x_test, batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ",RMSE(y_test, y_predict))                     



#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)