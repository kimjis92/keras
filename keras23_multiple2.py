from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import TensorBoard

def split_sequence(sequence, n_steps):
    X, y= list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence):
            break
        
        seq_x, seq_y = sequence[i:end_ix, :-1], sequence[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
        
    return array(X), array(y)



in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95, 105])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1.shape)
print(out_seq.shape)


in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

print(in_seq1.shape)
print(in_seq2.shape)
print(out_seq.shape)

dataset = np.hstack((in_seq1, in_seq2, out_seq))
n_steps = 3

print(dataset)

x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])




x = x.reshape(8, 6)

x = x.reshape(x.shape[0], x.shape[1], 1)

print(x)
print(x.shape)



x_train = x[:6]
x_test = x[6:]
y_train = y[:6]
y_test = y[6:]

print(out_seq)
print(in_seq1)


model=Sequential()
model.add(LSTM(64, activation='relu', input_shape=(6, 1)))
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


x_pre=array([[90, 95], [100, 105], [110, 115]])

x_pre=x_pre.reshape(1, 6, 1)

y_pre=model.predict(x_pre)

print(y_pre)
