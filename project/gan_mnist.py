#!/usr/bin/env python

import tensorflow as tf
tf.__version__

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import PIL
import time
import datetime
from IPython import display

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose as C2DT
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras.datasets import mnist


#TODO:

# Try dataset with colors (CIFAR10 almost done)
# Try dataset with higher resolution
# Create a dataset with only one category (shoes?)
# Create more samples (Flip, rotate, skew..)


# NOTES:
# Might want to use Keras own image preprocessing functions
# when taking in a whole directory of images in png format.
# see https://keras.io/api/preprocessing/image/.

# Grade system (loosely based on Goodwins feedback)
# C - Color and medium resolution, with a filled in report (Master template with some text on each chapter)
# B - High resolution, and proof of concept for further research (place new clothes on models)
# A - Innovative


# GPU workaround
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


### CONFIG ###
EPOCHS = 50
NOISE_DIM = 100
NUM_EXAMPLES = 16

BUFFER_SIZE = 40000
BATCH_SIZE = 64

DATA_PATH = "data"
IMAGE_PATH = "imgs"
MODELS_PATH = "models"
LOSS_PATH = "loss"


### INIT ###
seed = tf.random.normal([NUM_EXAMPLES, NOISE_DIM])
now = datetime.datetime.now().strftime("%d%H%M%S")
folder = os.path.join(IMAGE_PATH, now)
MODELS_FOLDER = os.path.join(MODELS_PATH, now)
LOSS_FOLDER = os.path.join(LOSS_PATH, now)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Make sure folders exists
if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)
if not os.path.exists(MODELS_FOLDER):
    os.mkdir(MODELS_FOLDER)
if not os.path.exists(folder):
    os.mkdir(folder)
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)
if not os.path.exists(LOSS_PATH):
    os.mkdir(LOSS_PATH)

def get_dataset():
    print("Getting Mnist dataset...")
    (images, labels), (_, _) = mnist.load_data()

#     print(labels)
#     temp = []
#     for image, label in zip(images, labels):
#         if label in [0,1,2,3]:
#             temp.append(image)
#     images = np.asarray(temp)

    print("Number of images:", images.shape)
    train_images = images.reshape(images.shape[0], 28, 28, 1).astype(np.float32)
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset


def make_generator_model():
    model = Sequential()
    init = RandomNormal(mean=0.0, stddev=0.02)

    model.add(Dense(7*7*128, kernel_initializer=init, use_bias=False, input_shape=(100,)))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(C2DT(64, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
#     model.add(C2DT(64, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
    
#     model.add(C2DT(32, (5, 5), strides=(2, 2), kernel_initializer=init, padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
    
    model.add(C2DT(1, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, activation='tanh'))
    
    print(model.output_shape)
    assert model.output_shape == (None, 28, 28, 1)
    return model


def make_discriminator_model():
    model = Sequential()
    
    model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

#     model.add(Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dropout(0.4))

    
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    G_optimizer.apply_gradients(zip(gradients_of_generator, 
        generator.trainable_variables))
    D_optimizer.apply_gradients(zip(gradients_of_discriminator, 
        discriminator.trainable_variables))
    return gen_loss, disc_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def train_forever(dataset):
    G_loss_list = []
    D_loss_list = []
    epoch = 0
    try:
        while(True):
            start = time.time()

            for image_batch in dataset:
                gen_loss, disc_loss = train_step(image_batch)
                G_loss_list.append(gen_loss)
                D_loss_list.append(disc_loss)

            if (epoch + 1) % 10 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)

                display.clear_output(wait=True)
                generate_and_save_images(generator, epoch + 1, seed)
#                 save_loss_curves(G_loss_list, D_loss_list)


            print ('epoch {: 3} took {:.4f} sec, with {:.4f} gen_loss and {:.4f} disc_loss'.format(epoch + 1, time.time()-start, gen_loss, disc_loss))
            epoch += 1
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Stopping training.")
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 2, seed)
        
        
def save_loss_curves(G_loss_list, D_loss_list):
    fig = plt.figure(figsize=(10,10))
    plt.plot(G_loss_list,color='red',label='Generator_loss')
    plt.plot(D_loss_list,color='blue',label='Discriminator_loss')
    plt.legend()
    plt.xlabel('total batches')
    plt.ylabel('loss')
    plt.title('Model loss per batch')
    plt.savefig("{}/loss_{}.png".format(LOSS_PATH, now))
    plt.close(fig)

    
def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('{}/image_at_epoch_{:05d}.png'.format(folder, epoch))
    plt.close(fig)

    # Save generator model
    filename = '{}/generator_model_{:05d}.h5'.format(MODELS_FOLDER, epoch)
    model.save(filename)


if __name__ == '__main__':

    train_dataset   = get_dataset()
    generator       = make_generator_model()
    discriminator   = make_discriminator_model()

    cross_entropy = BinaryCrossentropy()

    # learningrate started at 1e-4
    G_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    D_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(generator_optimizer=G_optimizer,
                                    discriminator_optimizer=D_optimizer,
                                    generator=generator, discriminator=discriminator)

    #train(train_dataset, EPOCHS)
    train_forever(train_dataset)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

