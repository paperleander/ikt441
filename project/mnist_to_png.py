#!/usr/bin/python

import os
import cv2

from tensorflow.keras.datasets import mnist

(train_images, _), (_,_) = mnist.load_data()

image_path = "./data/mnist"

if not os.path.exists(image_path):
    os.makedir(image_path)

for i, image in enumerate(train_images):
    cv2.imwrite(os.path.join(image_path, '{:05d}.png'.format(i)), image)
