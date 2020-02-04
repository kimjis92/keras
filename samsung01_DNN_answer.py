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
    
x, y = split_xy5(samsung, 5, 1)
print(x.shape)
print(y.shape)    
print(x[0, :], '\n', y[0])

# 데이터셋 나누기
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(
    x, y, random_state=1, test_size=0.3, shuffle=False)

print(x_train.shape)
print(x_test.shape)

# 데이터 전처리
# StandardSclaer
# 3차원 -> 2차원

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
print(x_train.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled[0, :])

# 모델구성
from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(128, activation='relu' ,input_shape=(25, )))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train_scaled, y_train, epochs=100, validation_split=0.25, batch_size=1)

eva=model.evaluate(x_test_scaled, y_test, batch_size=1)
print('loss : ', eva[0])
print('mse : ', eva[1])

y_pred = model.predict(x_test_scaled)

for i in range(5):
    print('종가 : ', y_test[i], '/ 예측가 : ', y_pred[i])

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ",RMSE(y_test, y_pred)) 