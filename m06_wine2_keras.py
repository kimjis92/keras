import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv', sep=';', encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

print(wine['quality'].value_counts())

enc=LabelEncoder()
y=enc.fit_transform(y)

from keras.utils import to_categorical

y=to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#2. 모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(11,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#3. 훈련
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
aaa= model.evaluate(x_test, y_test, batch_size=1)
print('aaa : ', aaa)

y_pred = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
