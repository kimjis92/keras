import numpy as np
import pandas as pd

df1=pd.read_csv('samsung.csv', index_col=0, header=0, encoding='cp949', sep=',')

print(df1)
print(df1.shape)


df2=pd.read_csv('kospi200.csv', index_col=0, header=0, encoding='cp949', sep=',')

print(df2)
print(df2.shape)


# kospi200의 모든 데이터
for i in range(len(df2.index)):
    df2.iloc[i,4]=int(df2.iloc[i,4].replace(',', ''))
    
    
# 삼성전자의 모든 데이터
for i in range(len(df1.index)):
    for j in range(len(df1.iloc[i])):
        df1.iloc[i,j]=int(df1.iloc[i,j].replace(',', ''))
        
        
print(df1)
print(df1.shape)


df1 = df1.sort_values(['일자'], ascending=[True])
df2 = df2.sort_values(['일자'], ascending=[True])

print(df2)

df1 = df1.values
df2 = df2.values

print(type(df1), type(df2))
print(df1.shape, df2.shape)

np.save('./data/samsung.npy', arr=df1)
np.save('./data/kospi200.npy', arr=df2)

samsung = np.load('./data/samsung.npy')
kospi200 = np.load('./data/kospi200.npy')

# print(samsung)
# print(samsung.shape)
# print(kospi200)
# print(kospi200.shape)

def split_xy5(dataset, time_steps, y_column):
    x, y= list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
    
        if y_end_number > len(dataset):
            break
        
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 3]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)
    
x1, y1 = split_xy5(samsung, 5, 1)
x2, y2 = split_xy5(kospi200, 5, 1)


# print(x.shape)
# print(y.shape)    
# print(x[0, :], '\n', y[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test=train_test_split(
    x1, y1, random_state=1, test_size=0.3, shuffle=False)
x2_train, x2_test, y2_train, y2_test=train_test_split(
    x2, y2, random_state=1, test_size=0.3, shuffle=False)

# print(x_train.shape)
# print(x_test.shape)

# 데이터 전처리
# StandardSclaer
# 3차원 -> 2차원

x1_train = np.reshape(x1_train, (x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2]))
x1_test = np.reshape(x1_test, (x1_test.shape[0], x1_test.shape[1] * x1_test.shape[2]))

x2_train = np.reshape(x2_train, (x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2]))
x2_test = np.reshape(x2_test, (x2_test.shape[0], x2_test.shape[1] * x2_test.shape[2]))


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x1_train)
x1_train_scaled = scaler.transform(x1_train)
x1_test_scaled = scaler.transform(x1_test)
x1_train_scaled = x1_train_scaled.reshape(x1_train_scaled.shape[0], 5, 5)
x1_test_scaled = x1_test_scaled.reshape(x1_test_scaled.shape[0], 5, 5)

print(x1_train_scaled[0, :])

scaler2=StandardScaler()
scaler2.fit(x2_train)
x2_train_scaled = scaler2.transform(x2_train)
x2_test_scaled = scaler2.transform(x2_test)
x2_train_scaled = x2_train_scaled.reshape(x2_train_scaled.shape[0], 5, 5)
x2_test_scaled = x2_test_scaled.reshape(x2_test_scaled.shape[0], 5, 5)
print(x2_train_scaled[0, :])

# 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
from keras.layers.merge import concatenate

input1=Input(shape=(5, 5))
dense1=LSTM(128)(input1)
dense2=Dense(64)(dense1)
dense3=Dense(32)(dense2)
dense4=Dense(16)(dense3)
dense5=Dense(8)(dense4)
dense6=Dense(4)(dense5)

output1=Dense(1)(dense6)

input2=Input(shape=(5, 5))
dense21=LSTM(128)(input2)
dense22=Dense(64)(dense21)
dense23=Dense(32)(dense22)
dense24=Dense(16)(dense23)
dense25=Dense(8)(dense24)
dense26=Dense(4)(dense25)

output2=Dense(1)(dense26)


merge1=concatenate([output1, output2])

middle1=Dense(32)(merge1)
middle2=Dense(16)(middle1)
middle3=Dense(8)(middle2)

output_1 = Dense(4)(middle3)
output_1 = Dense(1)(output_1)


model=Model(inputs=[input1, input2], outputs=[output_1])

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit([x1_train_scaled, x2_train_scaled], y1_train, epochs=100, validation_split=0.25, batch_size=1)

eva=model.evaluate([x1_test_scaled, x2_test_scaled], y1_test, batch_size=1)
print('loss : ', eva[0])
print('mse : ', eva[1])

y_pred = model.predict([x1_test_scaled, x2_test_scaled])

for i in range(5):
    print('종가 : ', y1_test[i], '/ 예측가 : ', y1_pred[i])

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y1_test, y_pred)) 