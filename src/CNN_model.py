#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import os
import cv2

from keras.datasets import mnist
from keras.utils import np_utils


from keras.models import model_from_json
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras import backend as K

K.set_image_data_format('channels_first')
 

#POUR LE TUTORIEL UNIQUEMENT
# fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)

###### PREPARE DATA ######
def extract_image_to_train(filename):
    filenames = os.listdir(filename)
    X_train_bonus = [] #= np.zeros([len(filenames)], dtype=object)
    y_train_bonus = np.zeros([len(filenames)], dtype=np.uint8)
    i = 0
    for f in filenames:
        X_train_bonus.append(cv2.imread(filename+f, 0))
        y_train_bonus[i] = np.uint8(f[0])
        i += 1
    X_train_bonus = np.asarray(X_train_bonus)
    return (X_train_bonus, y_train_bonus)


def get_and_prepare_data_mnist():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train_bonus, y_train_bonus) = extract_image_to_train("/home/hugo/Sudoku/Projet-Sudoku/train_set/")
    # Reshape X_train originally 60000x28x28 -> 60000x1x28x28
    # type float32 because pixels are normalized (/255)
    X_train = np.concatenate((X_train, X_train_bonus))
    y_train = np.concatenate((y_train, y_train_bonus))
    
    X_train = np.reshape(X_train, (X_train.shape[0], 1, 28, 28)).astype('float32')
    X_test = np.reshape(X_test, (X_test.shape[0], 1, 28, 28)).astype('float32')

    # Normalize from 0->255 to 0->1
    X_train = X_train/255
    X_test = X_test/255
        
    # one hot encode outputs
    # before y_train = label of number to detect ex: 3
    # after y_train = [0 0 0 1. 0 0 0 0 0 0]
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    return (X_train, y_train), (X_test, y_test), num_classes

def model_CNN(num_classes):
    model = Sequential()
    model.add(Conv2D(28, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # loss : function of the deviation between the AI results and expected results
    # optimizer : adam, used to update the CNN and diminuate the loss adaptive moment estimation
    # metrics : IS NOT USED for CNN model, meaningless here
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
	
# Evaluate a model using data and expected predictions
def print_model_error_rate(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Model score : %.2f%%" % (scores[1]*100))
    print("Model error rate : %.2f%%" % (100-scores[1]*100))


# This function saves a model on the drive using two files : a json and an h5
def save_keras_model(model, filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename+".h5")

# This function loads a model from two files : a json and a h5
# BE CAREFUL : the model NEEDS TO BE COMPILED before any use !
def load_keras_model(filename):
    # load json and create model
    json_file = open(filename+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename+".h5")
    return loaded_model

def train_and_save_model(filename, train_set, test_set, num_classes):
    (X_train, y_train) = train_set
    (X_test, y_test) = test_set
    model = model_CNN(num_classes)
    model.fit(X_train, y_train, validation_data=(X_test, y_test) ,batch_size=200, epochs=12)
    print_model_error_rate(model, X_test, y_test)
    save_keras_model(model, "model_detec_chiffre")

#(X_train, y_train), (X_test, y_test), num_classes = get_and_prepare_data_mnist()
#train_and_save_model("model_detec_chiffre", (X_train, y_train), (X_test, y_test), num_classes)
