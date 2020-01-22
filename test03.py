#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.utils
# Generate dummy data
x_train=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y_train=[[1],[2],[3],[4],[5],[1],[2],[3],[4],[5]]
x_train=np.array(x_train)
y_train=np.array(y_train)
y_train=keras.utils.to_categorical(y_train)

model = Sequential()
model.add(Dense(16, input_dim=1, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=32)


# In[12]:


np.argmax(model.predict([11]))


# In[13]:


np.argmax(model.predict([12]))


# In[14]:


np.argmax(model.predict([13]))


# In[ ]:




