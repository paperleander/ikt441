import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import fashion_mnist


def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('{}/image_at_epoch_{:05d}.png'.format(folder, epoch))
    plt.close(fig)
    #plt.show()

def get_mnist_dataset():
    print("Getting Mnist dataset...")
    (images, _), (_, _) = mnist.load_data()

    images = np.asarray(images)
    print("Number of images:", images.shape)
    train_images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    return train_images


def get_fashion_mnist_dataset():
    print("Getting Fashion Mnist dataset...")
    (train_images, train_labels), (_, _) = fashion_mnist.load_data()

    # Lets try to only train on one category in the fashion mnist dataset
    # (0=Tshirt/top, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat, 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot)
    category = 6
    images = []
    for image, label in zip(train_images, train_labels):
        if label == category:
            images.append(image)

    images = np.asarray(images)
    print("Number of images:", images.shape)
    train_images = images.reshape(images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    return train_images


def get_cifar10_dataset():
    print("Getting cifar10 dataset...")
    (train, _), (_, _) = cifar10.load_data()

    images = np.asarray(train)
    print("Number of images:", images.shape)
    train_images = images.reshape(images.shape[0], 32, 32, 3).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    return train_images


def get_dataset(dataset):
    if dataset == '':
        print("Please select a dataset. Exiting...")
        sys.exit()
    if dataset == 'mnist':
        get_mnist_dataset()
    if dataset == 'fashion_mnist':
        get_fashion_mnist_dataset()
    if dataset == 'cifar10':
        get_cifar10_dataset()

