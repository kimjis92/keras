import pandas as pd
import numpy as np
from keras.layers.merge import concatenate

saqm=pd.read_csv('samsung.csv', encoding='cp949')

saqm.head()

sam=saqm.sort_values(by=['일자'], ascending=True)
sam.head()
sam.to_csv('samreal.csv', encoding='euc-kr')
sam=pd.read_csv('samreal.csv', thousands = ',', encoding='euc-kr')
sam.head()

kos=pd.read_csv('kospi200.csv', encoding='cp949')
kos=kos.sort_values(by=['일자'], ascending=True)
kos.to_csv('kosreal.csv', encoding='euc-kr')
kos=pd.read_csv('kosreal.csv', thousands = ',', encoding='euc-kr')
kos.head()


from sklearn.preprocessing import MinMaxScaler

date=sam['일자']
siga=sam['시가']
goga=sam['고가']
jotga=sam['저가']
jongga=sam['종가']
geourae=sam['거래량']


ksiga=kos['시가']
kgoga=kos['고가']
kjotga=kos['저가']
kjongga=kos['현재가']
kgeourae=kos['거래량']


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split


z=5
i=0
start=0
x=[]
y=[]
kx=[]
ky=[]

while True :
    if z>425 :
        break
    if i == z : 
        y.append(geourae[i])
        ky.append(kgeourae[i])
        z=z+1
        start=start+1
        i=start
        continue
    
    

    x.append([siga[i], goga[i], jotga[i], jongga[i], geourae[i]])
    kx.append([ksiga[i], kgoga[i], kjotga[i], kjongga[i], kgeourae[i]])
        
    i=i+1
    
    
x=np.array(x)
y=np.array(y)

kx=np.array(kx)
ky=np.array(ky)

x=x.reshape(421, -1)
kx=kx.reshape(421, -1)

x_train, x_test, y_train, y_test, kx_train, kx_test, ky_train, ky_test=train_test_split(x, y, kx, ky, test_size=0.2)

print(kx_train.shape)

input1=Input(shape=(25,))
dense1=Dense(128)(input1)
dense2=Dense(64)(dense1)
dense3=Dense(32)(dense2)
dense4=Dense(16)(dense3)
dense5=Dense(8)(dense4)
dense6=Dense(4)(dense5)

output1=Dense(1)(dense6)

input2=Input(shape=(25, ))
dense21=Dense(128)(input2)
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
model.fit([x_train, kx_train], [y_train], epochs=100, validation_split=0.25, batch_size=1)
eva=model.evaluate([x_test, kx_test], [y_test], batch_size=1)

k=[61800, 61800,    60700,    60800, 14916555,    59400,    59400, 58300,    58800, 
   23664541,    59100,    59700,    58800, 59100, 16446102,    58800,    58800,    
   56800,    57200, 20821939]
k.append(57800)
k.append( 58400)
k.append(56400)
k.append(56400)
k.append(19749457)
k=np.array(k)
k=k.reshape(1, 25)

kospi=[303.77, 304.72, 301.71, 302.33, 86908, 294.98, 296.3, 291.3, 292.77, 130172, 294.38, 295.67, 292.45, 293.98, 85731, 293.27, 294.11, 287.09, 288.37, 101535, 290.24, 291.47
       , 284.53, 284.53, 101455]
kospi=np.array(kospi)
kospi=kospi.reshape(1,25)

t=model.predict([k, kospi])

print(t)


from sklearn.metrics import mean_squared_error
y_predict=model.predict([x_test, kx_test])

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ",RMSE(y_test, y_predict)) 