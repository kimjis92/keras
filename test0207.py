


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators
from keras.models import Sequential
from keras.layers import Dense
# scikit-learn 0.20.3 에서 31개
# scikit-learn 0.21.2 에서 40개중 4개만 돔

import warnings
warnings.filterwarnings('ignore')


sam = pd.read_excel('./data/삼성전자 0203-0206.xlsx', encoding='utf-8')
sam=sam.sort_values(by=['거래량'], ascending=False)


x = sam.loc[:1, ['시가', '고가', '저가']]
y = sam.loc[2:, '종가']
x = np.array(x)
y = np.array(y)
y = y.reshape(-1, 1)


model=Sequential()

model.add(Dense(8, input_shape=(3,)))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

print(x)
print(y)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x, y, epochs=50, batch_size=1)

k=np.array([60100, 61100, 59700])
k=k.reshape(1,3)
#print(k.shape)

#print(x.shape)
md=model.predict(k)

print(md)

# # classifier 알고리즘 모두 추출하기 ---(*1)
# warnings.filterwarnings('ignore')
# allAlgorithms = all_estimators(type_filter='classifier')

# print(allAlgorithms)
# print(len(allAlgorithms))
# print(type(allAlgorithms))

# for(name, algorithm) in allAlgorithms:
#     # 각 알고지름 객체 생성하기 ---(*2)
#     clf = algorithm()
    
#     # 학습하고 평가하기 ---(*3)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     print(name, '의 정답률 = ', accuracy_score(y_test, y_pred))
