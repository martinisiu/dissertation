
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
import time

def time_image_generation(generator,
                noise_input,
                noise_class,
                ):

    START_TIME = time.time()
    images = generator.predict([noise_input, noise_class])
    END_TIME = time.time()

    return END_TIME - START_TIME 

def plot_images(generator,
                noise_input,
                noise_class,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them

    For visualization purposes, generate fake images
    then plot them in a square grid

    Arguments:
        generator (Model): The Generator Model for fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name

    """
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict([noise_input, noise_class])
    print(model_name , " labels for generated images: ", np.argmax(noise_class, axis=1))
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')

def analyse_100_image_generations(generator, class_label=None, numberOfImagesEachGen=1, verbose=False):
    noise_input = np.random.uniform(-1.0, 1.0, size=[numberOfImagesEachGen, 100])
    step = 0

    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(numberOfImagesEachGen, 1)]
    else:
        noise_class = np.zeros((numberOfImagesEachGen, 10))
        noise_class[:,class_label] = 1
        step = class_label
    result = []
    for i in range(100):
        if verbose:
            print("Working on image generation: " + str(i))
        result.append(time_image_generation(generator,noise_input=noise_input,noise_class=noise_class))
    
    return result

"""
def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[8, 100])
    step = 0

    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 8)]
    else:
        noise_class = np.zeros((8, 10))
        noise_class[:,class_label] = 1
        step = class_label
    
    #print("TOTAL TIME FOR IMAGE GENERATION IS " + str(time_image_generation(generator,
    #noise_input=noise_input,
    #noise_class=noise_class)))

    
    plot_images(generator,
                noise_input=noise_input,
                noise_class=noise_class,
                show=True,
                step=step,
                model_name="test_outputs")
    """
    
def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    step = 0
    if class_label is None:
        num_labels = 10
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 16)]
    else:
        noise_class = np.zeros((16, 10))
        noise_class[:,class_label] = 1
        step = class_label
    
    plot_images(generator,
                noise_input=noise_input,
                noise_class=noise_class,
                show=False,
                step=step,
                model_name="test_outputs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Specify a specific digit to generate"
    parser.add_argument("-d", "--digit", type=int, help=help_)
    args = parser.parse_args()

    generator = load_model('cgan_mnist.h5')

    class_label = None
    if args.digit is not None:
        class_label = args.digit
    test_generator(generator, class_label)
    #total_time = analyse_100_image_generations(generator, class_label,verbose=True)
    #print("The total time it takes to generate 100 images is" + str(total_time))
    #print("Average time per a photo" + str(sum(total_time)/100))