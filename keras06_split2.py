import numpy as np
from sklearn.model_selection import train_test_split
x=np.array(range(1, 101))
y=np.array(range(1, 101))

x_tv, x_test, y_tv, y_test=train_test_split(x, y, test_size=0.2, shuffle=False)
x_train, x_validation, y_train, y_validataion=train_test_split(x_tv, y_tv, test_size=0.25, shuffle=False)



print("x_train", x_train)
print("x_validation", x_validation)
print("x_test", x_test)




# from keras.models import Sequential
# from keras.layers import Dense
# model=Sequential()

# model.add(Dense(5, input_shape=(1, )))
# model.add(Dense(2))
# model.add(Dense(3))
# model.add(Dense(1))

# model.summary()

# model.compile(loss='mse', optimizer='adam', metrics=['mae'])
# model.fit(x_train, y_train, epochs=20, batch_size=32)

# eval=model.evaluate(x_test, y_test,  steps=20)
# print(eval[1])