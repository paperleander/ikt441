from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import os
import datetime
import numpy as np


epochs = 1000
batch_size = 256
save_interval = epochs / 100

seed = np.random.normal(0, 1, (25, 100))  # Global seed for generating sample images
now = datetime.datetime.now().strftime("%d%H%M")
folder = os.path.join("imgs", now)
os.mkdir(folder)  # Now we can place images in different folders based on time

#TODO (or at least try):
# Change loss function (from min(log 1-D) to max(log D))
# Flip labels when training generator (real=fake, fake=real)
# Sample from Gaussian distribution
# Separate real and fake images in mini-batches.
# Make the lables noisy for the discriminator (randomly flip labels)
# Use SGD for discriminator while leaving ADAM for generator.
# Track the progress between G and D.

class GAN():
    def __init__(self):
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

        self.generator= self.build_generator()
        self.generator.compile(loss='binary_crossentropy',optimizer='adam')

        #Entire GAN
        z = Input(shape=(100,))
        img = self.generator(z)
        self.discriminator.trainable = False
        isFakeOrReal = self.discriminator(img)
        self.combined = Model(z,isFakeOrReal)
        self.combined.compile(loss='binary_crossentropy',optimizer='adam')



    def build_generator(self):
        #Noise shape = (100,)
        model = Sequential()
        model.add(Dense(256, input_shape=(100,), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(512,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1024,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(28*28*1, activation='tanh'))
        model.add(Reshape((28,28,1)))
        model.summary()

        noise = Input(shape=(100,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        #Image size (28,28,1)
        model = Sequential()
        model.add(Flatten(input_shape=(28,28,1)))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=(28,28,1))
        validity = model(img)

        return Model(img, validity)

    def train(self, seed, epochs, batch_size=128, save_interval=50):
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1 (Normalize)
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            #Discriminator
            idx = np.random.randint(0,X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0,1,(half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch,1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs,np.zeros((half_batch,1)))
            d_loss = np.add(d_loss_real,d_loss_fake)*0.5

            #Generator / Entire GAN

            noise = np.random.normal(0,1, (batch_size, 100))
            g_loss = self.combined.train_on_batch(noise,np.ones((batch_size,1)))
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_imgs(epoch, seed)

    def save_imgs(self, epoch, seed):
        r, c = 5, 5
        #import pdb;pdb.set_trace()
        #noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(seed)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("imgs/{}/mnist_{:04d}.png".format(now, epoch))
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(seed, epochs=epochs, batch_size=batch_size, save_interval=save_interval)

