#!/usr/bin/python

# HELPERS

import os
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image

def loadImages(path):
    image_files = [os.path.join(path, filename) for filename in os.listdir(path)]
    print(image_files[0])
    return image_files

def display(image):
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def preprocess_dep(data):
    dim = (128, 128)
    images = []
    labels = []
    for i in range(len(data)):
        img = cv2.imread(data[i], cv2.IMREAD_GRAYSCALE) 
        res = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        #gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
        images.append(res)
        path = data[i].split("/")
        image_label = path[-1]
        labels.append(image_label[0])
    return images, labels

def preprocess(data):
    dim = (128, 128)
    images = []
    labels = []
    for i in range(len(data)):
        img = image.load_img(data[i], target_size=(128,128,1), grayscale=True)
        img = image.img_to_array(img)
        img = img/255
        images.append(img)
        path = data[i].split("/")
        image_label = path[-1]
        labels.append(image_label[0])
    return images, labels

def plot_accuracy(history):
    #Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_loss(loss):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def write_model(model):
    #Writing the model
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
 
def load_model():
    # load json and create model
    from keras.models import model_from_json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    model = loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return model

