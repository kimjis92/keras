import numpy as np
from sklearn.model_selection import train_test_split
x1=np.array([range(1, 101), range(101, 201), range(301, 401)])
x2=np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])

y1=np.array([range(1, 101), range(101, 201), range(301, 401)])
y2=np.array([range(1001, 1101), range(1101, 1201), range(1301, 1401)])
y3=np.array([range(1, 101), range(101, 201), range(301, 401)])


# print(x1.shape)  #(2, 10)
# print(y2.shape)  #(2, 10)


x1=x1.transpose()
x2=x2.transpose()
y1=y1.transpose()
y2=y2.transpose()
y3=y3.transpose()

# x=x.reshape(100,2)
# y=y.reshape(100,2)


x1_tv, x1_test, x2_tv, x2_test, y1_tv, y1_test, y2_tv, y2_test, y3_tv, y3_test=train_test_split(x1, x2, y1, y2, y3, test_size=0.2, random_state=66 ,shuffle=False)
x1_train, x1_validation, x2_train, x2_validation, y1_train, y1_validataion, y2_train, y2_validataion, y3_train, y3_validataion=train_test_split(x1_tv, 
                                                x2_tv, y1_tv, y2_tv, y3_tv, test_size=0.25, random_state=66 , shuffle=False)



# print("x_train", x_train)
# print("x_validation", x_validation)
# print("x_test", x_test)




from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.merge import concatenate
#model=Sequential()



input1=Input(shape=(3,))
dense1=Dense(5)(input1)
dense2=Dense(2)(dense1)
dense3=Dense(3)(dense2)
output1=Dense(1)(dense3)

input2=Input(shape=(3,))
dense21=Dense(7)(input2)
dense22=Dense(4)(dense21)
dense23=Dense(3)(dense22)
output2=Dense(5)(dense23)

merge1=concatenate([output1, output2])

middle1=Dense(4)(merge1)
middle2=Dense(7)(middle1)
middle3=Dense(1)(middle2)

output_1 = Dense(30)(middle3)
output_1 = Dense(3)(output_1)

output_2 = Dense(300)(middle3)
output_2 = Dense(5)(output_2)
output_2 = Dense(3)(output_2)


output_3 = Dense(10)(middle3)
output_3 = Dense(3)(output_3)

model=Model(inputs=[input1, input2], outputs=[output_1, output_2, output_3])


# model.add(Dense(8, input_shape=(3,)))
# model.add(Dense(16))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=60, validation_data=([x2_validation, x2_validation], [y1_validataion, y2_validataion, y3_validataion]))



aaa=model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=20)
print('aaa: ', aaa)







x1_prd=np.array([[201, 202, 203], [402, 403, 405], [701, 702, 703]])
x2_prd=np.array([[301, 302, 303], [901, 902, 903], [801, 802, 803]])
x1_prd=x1_prd.transpose()
x2_prd=x2_prd.transpose()


a, b, c=model.predict([x1_prd, x2_prd], batch_size=1)
print(a)
print(b)
print(c)



y1_predict, y2_predict, y3_predict=model.predict([x1_test, x2_test], batch_size=1)
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y2_test, y3_test, y1_predict, y2_predict, y3_predict):
    return np.sqrt([mean_squared_error(y1_test, y1_predict), mean_squared_error(y2_test, y2_predict), mean_squared_error(y3_test, y3_predict)])
print("RMSE : ",RMSE(y1_test, y1_predict, y2_test, y2_predict, y3_test, y3_predict))                     



#R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
r2_y3_predict = r2_score(y2_test, y3_predict)


print("R2 : ", (r2_y1_predict+r2_y2_predict+r2_y3_predict)/3)
