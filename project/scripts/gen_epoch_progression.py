import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
#from tensorflow.keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

NOISE_DIM = 100
SIZE = 28
NUM_SEEDS = 6
NUM_MODELS = 8

#FOLDER_NAME = "27185605"
FOLDER_NAME = "28161548"
IMAGE_FOLDER = os.path.join("../imgs", FOLDER_NAME)
MODELS_FOLDER = os.path.join("../models", FOLDER_NAME)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Distribution of which models to make picture of (50 is epoch 500)
#MODEL_DIST = [0, 1, 10, 50, 100, 200, 500, 1000]
MODEL_DIST = [0, 60, 122, 183, 245, 306, 367, -1]


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


def generate_with_pillow(models, seeds):
    images = []
    for j, model_path in enumerate(models):
        model = keras.models.load_model(model_path)
        images.append(model(seeds, training=False))

    img_size = (SIZE * NUM_MODELS, SIZE * NUM_SEEDS)
    img = Image.new('L', img_size)


    for i in range(NUM_SEEDS):
        for j in range(NUM_MODELS):
            image = images[j][i]
            image = np.asarray(image)
            image = image[:, :, 0] * 127.5 + 127.5
            #print(image)
            #plt.imshow(image.astype(int))
            #plt.show()
#           image = image.reshape(56, 56)
            image = image.astype(np.uint8)
            im = Image.fromarray(image)
            #im.save('report/asd.jpg')
            img.paste(im, (j*SIZE, i*SIZE))

    img.save('../report/{}.png'.format(FOLDER_NAME))


if __name__ == '__main__':
    seeds = make_seed()
    models = get_models()
    #generate_images(models, seeds)
    generate_with_pillow(models, seeds)

