import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.utils.testing import all_estimators
# scikit-learn 0.20.3 에서 31개
# scikit-learn 0.21.2 에서 40개중 4개만 돔

import warnings
warnings.filterwarnings('ignore')

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv('./data/iris2.csv', encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, 'Name']
x = iris_data.loc[:, ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

# 학습 전용과 테스트 전용 분리하기
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, shuffle=True)

# 그리드 서치에서 사용할 매개 변수 ---(*1)
parameters = {'n_estimators': np.arange(100, 1000, step=100), 'max_depth': np.arange(0, 20, step=2)}
    

# 그리드 서치 ---(*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV( RandomForestClassifier(), parameters, cv=kfold_cv)
model.fit(x_train, y_train)
print('최적의 매개변수 = ', model.best_estimator_)

# 최적의 매개변수로 평가하기 ---(*3)
y_pred = model.predict(x_test)
print('최종 정답률 = ', accuracy_score(y_test, y_pred))