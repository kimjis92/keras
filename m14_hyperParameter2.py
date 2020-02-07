# 14를 카피, gridsearch로 변경#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float32') / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

def build_network(keep_prob = 0.5, optimizer = 'adam'):
    inputs = Input(shape=(28*28, ), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation = 'relu', name = 'hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation = 'relu', name = 'hidden3')(x)
    x = Dropout(keep_prob)(x)
    prideiction = Dense(10, activation='softmax', name = 'output')(x)
    model = Model(inputs = inputs, outputs = prideiction)
    model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_hyperparameter():
    batches = [10,20,30,40,50]
    optimizer = ['rmsprop', 'adam', 'adadelta']
    dropout = [0,1, 0,2, 0,3, 0,4, 0,5]
    return {'batch_size':batches, 'optimizer':optimizer, 'keep_prob':dropout}

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=1)

hypermarameters =  [
    {'batches' : [10,20,30,40,50], 'optimizer' : ['rmsprop'], 'keep_prob' : [0,1, 0,2, 0,3, 0,4, 0,5]},
    {'batches' : [10,20,30,40,50], 'optimizer' : ['adam'], 'keep_prob' : [0,1, 0,2, 0,3, 0,4, 0,5]},
    {'batches' : [10,20,30,40,50], 'optimizer' : ['adadelta'], 'keep_prob' : [0,1, 0,2, 0,3, 0,4, 0,5]}
    
]

from sklearn.model_selection import RandomizedSearchCV
search = GridSearchCV(estimator=model, hypermarameters, n_jobs=1, cv=3, verbose=1)

search.fit(x_train, y_train)
print (search.best_params_)