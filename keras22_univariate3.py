from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import TensorBoard


def split_sequence(sequence, n_steps):
    X, y= list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
        
    return array(X), array(y)

dataset = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

n_steps = 3

x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])

print(x)

x=x.reshape(x.shape[0], 3, 1)
x_train=x[:5]
x_test=x[5:]
y_train=y[:5]
y_test=y[5:]


model=Sequential()
model.add(LSTM(64, activation='relu', input_shape=(x_train.shape[1],1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse', 'accuracy'])
tb_hist=TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[tb_hist])


loss, mse, acc=model.evaluate(x_test, y_test, batch_size=1)


print('loss : ', loss)
print('mse : ', mse)
print('acc : ', acc)


x_pre=array([90, 100, 110])

x_pre=x_pre.reshape(1, 3, 1)
y_pre=model.predict(x_pre)

print(y_pre)