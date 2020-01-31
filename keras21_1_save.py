
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape
from keras.callbacks import EarlyStopping, TensorBoard



model=Sequential()
model.add(LSTM(128, activation='relu', input_shape=(3,1), return_sequences=True))
# model.add(LSTM(128, activation='relu', input_shape=(3,1)))
# model.add(Reshape((128, 1)))
model.add(LSTM(64, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(4, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

model.summary()


model.save('./save/savetest01.h5')
print('저장 잘 됐다.')