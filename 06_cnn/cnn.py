# Convolutional Neural Network
# now with extra Vim love


import random
import keras
import numpy as np
import matplotlib.pyplot as plt

from helpers import *

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# PATHS
# image_path = "/media/leander/2907-AF0D/food11"  # minnepenn, hehe
image_path = "/home/leander/uia/ikt441/06_cnn/food11_small"  # small version
training_path = os.path.join(image_path, "training")
validation_path = os.path.join(image_path, "validation")
evaluation_path = os.path.join(image_path, "evaluation")


# CONFIG
batch_size = 128
num_classes = 11
epochs = 12
img_rows, img_cols = 128,128
input_shape = (img_rows, img_cols, 1)

# PREPROCESS
x_train, y_train = preprocess(loadImages(training_path))
x_test, y_test = preprocess(loadImages(validation_path)) 
x_eval, y_eval = preprocess(loadImages(evaluation_path)) 
x_train = np.array(x_train)
x_test = np.array(x_train)
x_eval = np.array(x_train)

# Print some of the images
for x in range(100):
    n = random.randint(0,len(x_train) - 1)
    plt.subplot(10,10,x+1)
    plt.axis('off')
    plt.imshow(x_train[n].reshape(img_rows, img_cols),cmap='gray')
plt.show()

# Keras is picky on the input shape..
x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
x_eval = x_eval.reshape(x_eval.shape[0],img_rows,img_cols,1)

# Convert uint8 gray pixels to floats (0-1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_eval = x_eval.astype('float32')
x_train /= 255
x_test /= 255
x_eval /= 255

# Convert labels to categorical one-hot encoding, since this is multi-class
y_train = keras.utils.to_categorical(y_train, num_classes=11)
y_test = keras.utils.to_categorical(y_test, num_classes=11)
y_eval = keras.utils.to_categorical(y_eval, num_classes=11)

# INFO
print('x_train shape:',x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# MAKE MODEL
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape = input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test,y_test))

score = model.evaluate(x_eval, y_eval,verbose=0)
print(model.summary())
print("Score:", score)
plot_accuracy(history)

