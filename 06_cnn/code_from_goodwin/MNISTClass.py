from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import random
import keras
batch_size = 128
num_classes = 10
epochs = 12
maxnum = 100

img_rows, img_cols = 28,28
(x_train,y_train),(x_test,y_test) = mnist.load_data()

import matplotlib.pyplot as plt

for x in range(100):
        n = random.randint(0,len(x_train))
        plt.subplot(10,10,x+1)
        plt.axis('off')
        plt.imshow(x_train[n].reshape(28,28),cmap='gray')
plt.show()

x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test = x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
input_shape = (img_rows,img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('x_train shape:',x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu", input_shape = input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(num_classes,activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy, 
                optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])


history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

score = model.evaluate(x_test,y_test,verbose=0)
print(score)
print(model.summary())

import numpy as np
print(model.predict(np.expand_dims(x_test[0],axis=0)).argmax)

#print(model.predict(np.array(x_test[0],)))

def plotEvaluation():
    (x_train_new,y_train_new),(x_test_new,y_test_new) = mnist.load_data()
    currentNumber = 0
    currentCount = 0
    x_train_new = x_train_new[:maxnum]
    y_train_new = y_train_new[:maxnum]


    for x in range(100):
        n = random.randint(0,len(x_train_new)-1)
        while(not y_train_new[n]==currentNumber):
            n = random.randint(0,len(x_train_new)-1)
        currentCount += 1
        if currentCount==10:
            currentCount=0
            currentNumber += 1
        plt.subplot(10,10,x+1)
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.axis('off')
        
        plt.imshow(x_train_new[n].reshape(28,28),cmap='gray')
        res = model.predict(np.expand_dims(x_train[n],axis=0)).argmax()
        plt.annotate(str(res),xy=(0,0),size=15,color="red")
    plt.show()


plotEvaluation()

#Writing the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



#Loading the model
# load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


