#! /usr/bin/env python3
# coding: utf-8

from CNN_model import load_keras_model
import traitement_image as TI
from sudoku_grid import Sudoku_grid
import sys
import numpy as np

model = load_keras_model("model_detec_chiffre")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def reshape_array(array):
    return np.reshape(array, (1, 1, 28, 28))

def final_process(filename):
    list_of_numbers = []
    list_of_numbers = TI.image_process(filename)
    sudoku_array = np.zeros([9,9])
    for i in range(len(list_of_numbers)):
        index = list_of_numbers[i][1]
        value = model.predict_classes(reshape_array(list_of_numbers[i][0]), verbose=1)
        sudoku_array[index] = value
    print(sudoku_array)


def main():
    final_process(str(sys.argv[1]))

if __name__ == "__main__":
    main()