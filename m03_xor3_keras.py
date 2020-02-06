from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import numpy as np


#1. 데이터
x_train = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_train = [0, 1, 1, 0]
x_train = np.array(x_train)
# x_train = x_train.transpose()

#2. 모델
# model = LinearSVC()
# model = KNeighborsClassifier(n_neighbors=1)
model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(2, )))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#3. 훈련
model.fit(x_train, y_train, batch_size=1, epochs=100,)


#4. 평가예측
x_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_predict = model.predict(x_test)


print(x_test, "의 예측결과 : ", y_predict)

