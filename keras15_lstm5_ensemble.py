from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input, Add
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate


x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5] ,[4, 5, 6], [5, 6, 7], [6, 7,8], [7, 8, 9], [8, 9, 10], [9, 10, 11]
           , [10, 11, 12], [20, 30 ,40], [30, 40, 50], [40, 50, 60]])
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50] ,[40, 50, 60], [50, 60, 70], [60, 70,80], [70, 80, 90], [80, 90, 100], [90, 100, 110]
           , [100, 110, 120], [2, 3 ,4], [3, 4, 5], [4, 5, 6]])
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])
# for i, xs, ys in zip(enumerate(x), y) :
#     if (xs[1]-xs[0])>1 :
#         x[i][1]=x[i][0]+1
#         x[i][2]=x[i][0]+2
#     elif (xs[1]-xs[0])<-1 :
#         x[i][1]=x[i][0]+1
#         x[i][2]=x[i][0]+2
        


x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)  #lstm에 넣기위해 x의 구조를 바꿈
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)  #lstm에 넣기위해 x의 구조를 바꿈



# model=Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(3,1)))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(2))
# model.add(Dense(1))

input1=Input(shape=(x1.shape[1],1))
lstm1=LSTM(64)(input1)
lstm2=Dense(32)(lstm1)
lstm3=Dense(16)(lstm2)
lstm4=Dense(8)(lstm3)
output1=Dense(4)(lstm4)


input2=Input(shape=(x1.shape[1],1))
lstm01=LSTM(64)(input2)
lstm02=Dense(32)(lstm01)
output2=Dense(4)(lstm02)


merge1=Add()([output1, output2])

sub1=Dense(8)(merge1)
sub2=Dense(4)(sub1)
sub3=Dense(1)(sub2)


sub01=Dense(4)(merge1)
sub02=Dense(1)(sub01)



model=Model(inputs=[input1, input2], outputs=[sub3, sub02])

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
early_stopping=EarlyStopping(monitor='acc',patience=30, mode='max')

model.fit([x1, x2], [y1, y2], epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping])



aaa=model.evaluate([x1, x2], [y1, y2], batch_size=1)


print('aaa : ', aaa)


x1_input=array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])
x1_input=x1_input.reshape(4, 3, 1)

# x2_input=array([[4.5, 5.5, 6.5], [30, 40, 50], [60, 70, 80], [120, 130, 140]])
x2_input=array([[6.5, 7.5, 8.5], [50, 60, 70], [70, 80, 90], [100, 110, 120]])
x2_input=x1_input.reshape(4, 3, 1)


y_predict=model.predict([x1_input, x2_input])
print(y_predict)