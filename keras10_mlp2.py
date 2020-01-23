import numpy as np
from sklearn.model_selection import train_test_split
x=np.array([range(1, 101), range(101, 201), range(301, 401)])
y=np.array([range(1, 101)])


print(x.shape)  #(2, 10)
print(y.shape)  #(2, 10)


x=x.transpose()
y=y.transpose()
# x=x.reshape(100,2)
# y=y.reshape(100,2)

print(x)


   
x_tv, x_test, y_tv, y_test=train_test_split(x, y, test_size=0.2, shuffle=False)
x_train, x_validation, y_train, y_validataion=train_test_split(x_tv, y_tv, test_size=0.25, shuffle=False)



print("x_train", x_train)
print("x_validation", x_validation)
print("x_test", x_test)




from keras.models import Sequential
from keras.layers import Dense
model=Sequential()

model.add(Dense(8, input_shape=(3,)))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=60, validation_data=(x_validation,y_validataion))



loss, mse, val_acc=model.evaluate(x_test, y_test,  batch_size=20)
print('val_acc : ', val_acc)

x_prd=np.array([[501], [204], [304]])
x_prd=x_prd.transpose()

aaa=model.predict(x_prd, batch_size=1)
print(aaa)


y_predict=model.predict(x_test, batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ",RMSE(y_test, y_predict))                     



#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
