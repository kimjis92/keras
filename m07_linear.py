from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
boston = load_boston()

x = boston.data
y = boston.target

from sklearn.linear_model import LinearRegression, Ridge, Lasso

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

# LinearRegression 모델

lin = LinearRegression()

lin.fit(x_train, y_train)

y_pred = lin.predict(x_test)

# print('정답률 : ', accuracy_score(y_test, y_pred))
print('LinearRegression r2 값 : ', r2_score(y_test, y_pred))



# Ridge 모델

rid = Ridge()

rid.fit(x_train, y_train)

y_pred = rid.predict(x_test)

print('Ridge r2 값 : ', r2_score(y_test, y_pred))

# Lasso 모델

las = Lasso()

las.fit(x_train, y_train)

y_pred = las.predict(x_test)

print('Lasso r2 값 : ', r2_score(y_test, y_pred))