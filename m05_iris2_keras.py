
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

#1. 데이터 붓꽃 데이터 읽어들이기
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8', names=['a', 'b', 'c', 'd', 'y'])




# 붗꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, 'y']
x = iris_data.loc[:, ['a', 'b', 'c', 'd']]

y = np.array(y)
x = np.array(x)

# from sklearn.preprocessing import LabelEncoder

# enc=LabelEncoder()
# y=enc.fit_transform(y)

# from keras.utils import to_categorical

# y=to_categorical(y)

# print(y)


one=OneHotEncoder()

k=[['Iris-setosa'], ['Iris-versicolor'], ['Iris-virginica']]
one.fit(k)
y=y.reshape(-1 ,1)

y=one.transform(y)
y=y.toarray()
# y=to_categorical(y)
# y=y.reshape(-1, 3)
print(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7, shuffle=True)


# # 학습하기
# model = Sequential()
# model.add(Dense(32, activation='relu', input_shape=(4,)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# model.fit(x_train, y_train, epochs=100, batch_size=1)

# # 평가하기
# y_pred = model.predict(x_test)
# print("정답률 : ", accuracy_score(y_test, y_pred)) # 0.933 ~ 1.0






