import matplotlib.pyplot as plt
import scipy.misc
import os
import numpy as np
# from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

NOISE_DIM = 100
SIZE = 56
NUM_HEIGHT = 6
NUM_WIDTH = 6

# WIDTH has to be larger than HEIGHT, and I might have mixed them up...
NUM_SEEDS = NUM_HEIGHT * NUM_WIDTH
NUM_MODELS = NUM_SEEDS

# FOLDER_NAME = "27185605"
FOLDER_NAME = "23144152"
IMAGE_FOLDER = os.path.join("../imgs", FOLDER_NAME)
MODELS_FOLDER = os.path.join("../models", FOLDER_NAME)

# Distribution of which models to make picture of (50 is epoch 500)
MODEL_DIST = [0, 30, 60, 90, 120, 150, 200, 100]
# MODEL_DIST = [0, 60, 122, 183, 245, 306, 367, -1]

if not os.path.exists("../report"):
    os.mkdir("../report")


def make_seed():
    return tf.random.normal([NUM_SEEDS, NOISE_DIM])


def get_models():
    model_paths = os.listdir(MODELS_FOLDER)
    model_paths.sort()
    models = []
    for i in MODEL_DIST:
        models.append(os.path.join(MODELS_FOLDER, model_paths[i]))
    print("MODELS")
    print(models)
    return models


def generate_with_pillow(models, seeds, color_channels, sample=False):
    if sample:
        images = sample_images()
    else:
        model = keras.models.load_model(models[-1])  # last models is the best
        images = model(seeds, training=False)

    img_size = (SIZE * NUM_HEIGHT, SIZE * NUM_WIDTH)

    if color_channels == 3:
        img = Image.new('RGB', img_size)
        for i in range(NUM_HEIGHT):
            for j in range(NUM_WIDTH):
                pos = i * NUM_HEIGHT + j
                image = images[pos]
                image = np.asarray(image)
                image = image[:, :, :] * 127.5 + 127.5
                #             image = image[:, :, :] * 127.5 + 127.5
                #             print(image.astype(int))
                #             print(image.shape)
                #             image = image.reshape((SIZE, SIZE))
                image = image.astype(np.uint8)
                im = Image.fromarray(image)
                img.paste(im, (i * SIZE, j * SIZE))
    else:
        img = Image.new('L', img_size)
        for i in range(NUM_HEIGHT):
            for j in range(NUM_WIDTH):
                pos = i * NUM_HEIGHT + j
                image = images[pos]
                image = np.asarray(image)
                image = image[:, :, 0] * 127.5 + 127.5
                image = image.astype(np.uint8)
                im = Image.fromarray(image)
                img.paste(im, (i * SIZE, j * SIZE))

    img.save('../report/collage_{}x{}_{}.png'.format(NUM_HEIGHT, NUM_WIDTH, FOLDER_NAME))


def get_dataset():
    shoes_path = "../data/fashion-shoes-56"
    shoes = os.listdir(shoes_path)
    images = []
    for k in shoes:
        if k.startswith("."):
            continue
        img_path = os.path.join(shoes_path, k)

        # img = image.load_img(img_path, target_size=(28,28,3), color_mode="rgb")
        # Reference: https://github.com/carpedm20/DCGAN-tensorflow/issues/162#issuecomment-315519747
        img_bgr = cv2.imread(img_path)
        # Reference: https://stackoverflow.com/a/15074748/
        img_rgb = img_bgr[..., ::-1]  # bgr2rgb
        img = image.img_to_array(img_rgb)
        images.append(img)

    images = np.asarray(images)
    print("Number of images:", images.shape)
    train_images = images.reshape((images.shape[0], 56, 56, 3)).astype(np.float32)
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    return train_images


def get_dataset_mnist():
    print("Getting Mnist dataset...")
    (images, labels), (_, _) = mnist.load_data()

    print("Number of images:", images.shape)
    train_images = images.reshape(images.shape[0], 28, 28, 1).astype(np.float32)
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return train_images


def sample_images():
    # images = get_dataset()
    images = get_dataset_mnist()
    idx = np.random.randint(images.shape[0], size=36)
    return images[idx, :, :]


if __name__ == '__main__':
    seeds = make_seed()
    models = get_models()
    # generate_images(models, seeds)
    generate_with_pillow(models, seeds, color_channels=3, sample=False)
