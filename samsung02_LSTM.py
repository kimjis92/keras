import pandas as pd
import numpy as np

saqm=pd.read_csv('samsung.csv', encoding='cp949')

saqm.head()

sam=saqm.sort_values(by=['일자'], ascending=True)
sam.head()
sam.to_csv('samreal.csv', encoding='euc-kr')
sam=pd.read_csv('samreal.csv', thousands = ',', encoding='euc-kr')
sam.head()

from sklearn.preprocessing import MinMaxScaler

date=sam['일자']
siga=sam['시가']
goga=sam['고가']
jotga=sam['저가']
jongga=sam['종가']
geourae=sam['거래량']

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split


z=5
i=0
start=0
x=[]
y=[]


while True :
    if z>425 :
        break
    if i == z : 
        y.append(geourae[i])
        z=z+1
        start=start+1
        i=start
        continue

    x.append([siga[i], goga[i], jotga[i], jongga[i], geourae[i]])    
    i=i+1
    
    
x=np.array(x)
y=np.array(y)

x=x.reshape(421, 5, 5)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2)

model=Sequential()
model.add(LSTM(128, activation='relu' ,input_shape=(5, 5)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=40, validation_split=0.25, batch_size=1)
eva=model.evaluate(x_test, y_test, batch_size=1)

k=[61800, 61800,    60700,    60800, 14916555,    59400,    59400, 58300,    58800, 
   23664541,    59100,    59700,    58800, 59100, 16446102,    58800,    58800,    
   56800,    57200, 20821939]
k.append(57800)
k.append( 58400)
k.append(56400)
k.append(56400)
k.append(19749457)
k=np.array(k)
k=k.reshape(1,5, 5)

t=model.predict(k)

print(t)


from sklearn.metrics import mean_squared_error
y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ",RMSE(y_test, y_predict)) 