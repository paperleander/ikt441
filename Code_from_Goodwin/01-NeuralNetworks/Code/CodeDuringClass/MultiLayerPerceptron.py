import numpy as np
import math
import sys

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import matplotlib.pyplot as plt

# crim	tax	rm	    age	    ptratio	medv
training_dataset = """
0.00632	296	6.575	65.2	15.3	24
0.02731	242	6.421	78.9	17.8	21.6
0.03237	222	6.998	45.8	18.7	33.4
0.06905	222	7.147	54.2	18.7	36.2
0.08829	311	6.012	66.6	15.2	22.9
0.22489	311	6.377	94.3	15.2	15
0.11747	311	6.009	82.9	15.2	18.9
0.09378	311	5.889	39	15.2	21.7
0.62976	307	5.949	61.8	21	20.4
"""

training_dataset = [[float(f) for f in i.split("\t")] for i in training_dataset.strip().split("\n")]
training_dataset = [row[:-1]+[0 if row[-1] < 20 else 1] for row in training_dataset]
training_dataset = training_dataset*100

X = np.array([i[0:5] for i in training_dataset])
Y = np.array([i[5] for i in training_dataset])

np.random.seed(7)

model = Sequential()
model.add(Dense(5, input_dim=len(X[0])))
model.add(Activation("sigmoid"))
model.add(Dense(2))
model.add(Activation("sigmoid"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, Y, epochs=1000)

model.summary()

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

print(training_dataset)

weights = [-0.1, 0.20, -0.23, -0.1, 0.20, -0.23, -0.1, 0.20, -0.23]

# sys.exit(0)


def sigmoid(z):
    return 1/(1 + math.exp(-z))


def first_layer(row, weights):
    # First neuron
    a_1 = weights[0]*1
    a_1 += weights[1]*row[0]
    a_1 += weights[2]*row[1]

    # Second neuron
    a_2 = weights[3]*1
    a_2 += weights[4]*row[2]
    a_2 += weights[5]*row[3]

    return sigmoid(a_1), sigmoid(a_2)


def second_layer(row, weights):
    activation_3 = weights[6]
    activation_3 += weights[7]*row[0]
    activation_3 += weights[8]*row[1]
    return sigmoid(activation_3)


def predict(row, weights):
    fl = first_layer(row, weights)
    sl = second_layer(fl, weights)
    return fl, sl


for d in training_dataset:
    print(predict(d, weights)[1], d[-1])


def train_weights(train, learningrate, epochs):
    # weights = [random.uniform(-1,1) for i in range(len(train[0]))]
    last_error = 0.0
    for epoch in range(epochs):
        sum_error = 0.0
        for row in train:
            _first_layer, prediction = predict(row, weights)
            error = row[-1]-prediction
            # print(error)
            sum_error += error**2  # abs(error)#math.abs(error)#**2**0.5

            # First layer
            weights[0] = weights[0] + learningrate*error
            weights[1] = weights[1] + learningrate*error*row[0]
            weights[2] = weights[2] + learningrate*error*row[1]

            weights[3] = weights[3] + learningrate*error
            weights[4] = weights[4] + learningrate*error*row[2]
            weights[5] = weights[5] + learningrate*error*row[3]

            # Second layer
            weights[6] = weights[6] + learningrate*error
            weights[7] = weights[7] + learningrate*error*_first_layer[0]
            weights[8] = weights[8] + learningrate*error*_first_layer[1]
            
            # for i in range(len(row)-1):
            #    weights[i+1] = weights[i+1] + learningrate*error*row[i]
        if(epoch % 100 == 0) or (last_error != sum_error):
            print("Epoch "+str(epoch) + " Learning rate " + str(learningrate) + " Error " + str(sum_error))
        last_error = sum_error
    return weights


learningrate = 0.0001
epochs = 1000

train_weights = train_weights(training_dataset, learningrate, epochs)
print(train_weights)
