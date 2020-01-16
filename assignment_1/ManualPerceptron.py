"""

In this task you are suppose to implement 2 types of multilayer Perceptrons:
1. Using only Python.
2. Using a high level library
•Download the Ecoli dataset:https://archive.ics.uci.edu/ml/datasets/Ecoli
•Predict the two classes: cp and im (remove the rest of the dataset).
•Make the necessary adjustments to the data.
•Implement and test a Multilayer Perceptron from scratch using only Python and standard libraries.
•Implement and test a Multilayer Perceptron using a high level library (e.g.,Keras, TensorFlow, Torch).
•Choose the network architecture with care.
•Train and validate all algorithms.
•Make the necessary assumptions
•Write a one-page report about the assignment

"""

import math
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_table("data/ecoli.data",
                   header=None,
                   delim_whitespace=True,
                   names=["seq", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "label"])

# df.hist(figsize=(15, 12))
# plt.show()

df_clean = df[(df.label == "im") | (df.label == "cp")]
# print(df_clean)

X = df_clean.drop(['label', 'seq'], axis=1).to_numpy()
y = df_clean.label.replace({'cp': 0, 'im': 1}).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def first_layer(row, weights):
    # First neuron
    a_1 = weights[0] * 1
    a_1 += weights[1] * row[0]
    a_1 += weights[2] * row[1]

    # Second neuron
    a_2 = weights[3] * 1
    a_2 += weights[4] * row[2]
    a_2 += weights[5] * row[3]
    a_2 += weights[6] * row[4]

    # Third neuron
    a_3 = weights[7] * 1
    a_3 += weights[8] * row[5]
    a_3 += weights[9] * row[6]

    return sigmoid(a_1), sigmoid(a_2), sigmoid(a_3)


def second_layer(row, weights):
    activation_3 = weights[10]
    activation_3 += weights[11] * row[0]
    activation_3 += weights[12] * row[1]
    activation_3 += weights[13] * row[2]
    return 1 if activation_3 > 0 else 0
    # return sigmoid(activation_3)


def predict(row, weights):
    fl = first_layer(row, weights)
    sl = second_layer(fl, weights)
    return fl, sl


def train_weights(train, Y, learningrate, epochs):
    weights = [random.uniform(-1, 1) for _ in range(len(train[0]) + 7)]
    df_epochs = []
    errors = []
    # plus 7 because of 3 extra weights at first layer and 4 extra weights at second layer
    for epoch in range(epochs):
        sum_error = 0.0
        for row, y in zip(train, Y):
            _first_layer, _prediction = predict(row, weights)
            error = y - _prediction

            sum_error += error ** 2

            # First layer
            weights[0] = weights[0] + learningrate * error
            weights[1] = weights[1] + learningrate * error * row[0]
            weights[2] = weights[2] + learningrate * error * row[1]

            weights[3] = weights[3] + learningrate * error
            weights[4] = weights[4] + learningrate * error * row[2]
            weights[5] = weights[5] + learningrate * error * row[3]
            weights[6] = weights[6] + learningrate * error * row[4]

            weights[7] = weights[7] + learningrate * error
            weights[8] = weights[8] + learningrate * error * row[5]
            weights[9] = weights[9] + learningrate * error * row[6]

            # Second layer
            weights[10] = weights[10] + learningrate * error
            weights[11] = weights[11] + learningrate * error * _first_layer[0]
            weights[12] = weights[12] + learningrate * error * _first_layer[1]
            weights[13] = weights[13] + learningrate * error * _first_layer[2]

        if epoch % 100 == 0:
            print("Epoch " + str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))

        errors.append(sum_error)
        df_epochs.append(epoch)

    return df_epochs, errors, weights


learningrate = 0.001
epochs = 10000

df_epochs, errors, train_weights = train_weights(X_train, y_train, learningrate, epochs)
print("Weights:", train_weights)

accuracies = 0
for x, y in zip(X_test, y_test):
    _, prediction = predict(x, train_weights)
    accuracies += 1 if prediction == y else 0
    # print("Expected vs Real:", y, prediction)

accuracy = float(accuracies/len(X_test) * 100)
print("Accuracy", accuracy)

plt.plot(df_epochs, errors)
title = "Sum of errors over " + str(epochs) + " epochs with " + str(learningrate) + " learningrate"
plt.title(title)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()
