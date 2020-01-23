import numpy as np
x=np.array(range(1, 101))
y=np.array(range(1, 101))

x_train=x[:61]
x_validation=x[61:81]
x_test=x[81:]

y_train=y[:61]
y_validation=y[61:81]
y_test=y[81:]

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