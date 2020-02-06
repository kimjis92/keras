from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf


# seed 값 생성
seed=0
np.random.seed(seed)
tf.set_random_seed(seed)

# 데이터 로드
dataset = np.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8]
Y = dataset[:, 8]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# 모델의 설정
model = KNeighborsClassifier(n_neighbors=1)


# 모델 실행
model.fit(X, Y)


y_predict = model.predict(x_test)

# 결과 출력

print('y_predict : ', y_predict)
print("acc : ", accuracy_score(y_test , y_predict))

