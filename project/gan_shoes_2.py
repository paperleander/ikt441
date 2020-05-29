#!/usr/bin/env python
# NB!!!! NEEDS TWEAKING

import tensorflow as tf
print(tf.__version__)

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import datetime
from IPython import display

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2DTranspose as C2DT
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image

import cv2

#TODO:
# Best architecture on 28x28 black/white
# Colors on 28x28
# Higher Resolution (56x56) on black/white
# Higher Resolution (56x56) on color
# Load older models
# Make a "timeline" of epoch progression

# NOTES:
# Might want to use Keras own image preprocessing functions
# when taking in a whole directory of images in png format.
# see https://keras.io/api/preprocessing/image/.

# Grade system (loosely based on Goodwins feedback)
# C - Color and medium 28, with a filled in report (Master template with some text on each chapter)
# B - High 28, and proof of concept for further research (place new clothes on models)
# A - Innovative


# GPU workaround
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


### CONFIG ###
EPOCHS = 50
NOISE_DIM = 100
NUM_EXAMPLES = 16

BUFFER_SIZE = 40000
BATCH_SIZE = 128

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
    shoes_path = "data/fashion-shoes-sport-casual-28"
    shoes = os.listdir(shoes_path)
    images = []
    for k in shoes:
        if k.startswith("."):
            continue
        img_path = os.path.join(shoes_path, k)

        #img = image.load_img(img_path, target_size=(28,28,3), color_mode="rgb")
        # Reference: https://github.com/carpedm20/DCGAN-tensorflow/issues/162#issuecomment-315519747
        img_bgr = cv2.imread(img_path)
        # Reference: https://stackoverflow.com/a/15074748/
        img_rgb = img_bgr[..., ::-1] # bgr2rgb
        img = image.img_to_array(img_rgb)
        images.append(img)

    images = np.asarray(images)

    print("Number of images:", images.shape)
    train_images = images.reshape(images.shape[0], 28, 28, 3).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    print("min val: {}, max val: {}".format(np.min(train_images[0]), np.max(train_images[0])))
    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32, 5, strides=2, padding='same', input_shape=(28, 28, 3)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(64, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(128, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Conv2D(256, 5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    return model

def make_generator_model():
    model = tf.keras.Sequential()
    init = RandomNormal(mean=0.0, stddev=0.02)

    model.add(Dense(7*7*128, kernel_initializer=init, input_shape=(100,)))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(C2DT(64, 5, strides=2, padding='same', kernel_initializer=init))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
              
#     model.add(C2DT(32, 5, strides=2, padding='same', kernel_initializer=init))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.2))
              
    model.add(C2DT(3, 5, strides=2, padding='same', activation='tanh', kernel_initializer=init))
    
    print(model.summary())
              
    assert model.output_shape == (None, 28, 28, 3)
    return model


def discriminator_loss(real_output, fake_output):
    # Might want to implement "Label smoothing"
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = (real_loss) + fake_loss
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


# def train(dataset, epochs):
#     for epoch in range(epochs):
#         start = time.time()

#         for image_batch in dataset:
#             gen_loss, disc_loss = train_step(image_batch)

#         display.clear_output(wait=True)
#         generate_and_save_images(generator, epoch + 1, seed)

#         if (epoch + 1) % 15 == 0:
#             checkpoint.save(file_prefix = checkpoint_prefix)

#         print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#     display.clear_output(wait=True)
#     generate_and_save_images(generator, epochs, seed)


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
                save_loss_curves(G_loss_list, D_loss_list)


            print ('epoch {: 3} took {:.4f} sec, with {:.4f} gen_loss and {:.4f} disc_loss'.format(epoch + 1, time.time()-start, gen_loss, disc_loss))
            epoch += 1
    except KeyboardInterrupt:
        print("KeyboardInterrupt. Stopping training.")
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 2, seed)

              
def save_loss_curves(G_loss_list, D_loss_list):
    plt.figure(figsize=(10,10))
    plt.plot(G_loss_list,color='red',label='Generator_loss')
    plt.plot(D_loss_list,color='blue',label='Discriminator_loss')
    plt.legend()
    plt.xlabel('total batches')
    plt.ylabel('loss')
    plt.title('Model loss per batch')
    plt.savefig("{}/loss_{}.png".format(LOSS_PATH, now))
              
              
def generate_and_save_images(model, epoch, seed):
    predictions = model(seed, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255.)
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
                                    generator=generator, 
                                    discriminator=discriminator)

    #train(train_dataset, EPOCHS)
    train_forever(train_dataset)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
