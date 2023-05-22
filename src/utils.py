'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, FINAL PROJECT: Waste classification using CNN's

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
This script contains plotting functions which are used in the other two scripts.

'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input)

def save_plot_history(H, epochs, name):
    '''
    Saves the validation and loss history plots of a fitted model.
    Code is adapted from the Session 9 notebook of the Visual Analytics course at AU, 2023.
    
    Arguments:
    - H: History of a model fit
    - Epochs: Number of epochs the model runs on
    - name: prefix to the filename the plot will be saved as
    
    Returns:
      None
    '''
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join("out", f"{name}_history_plot.png"))

def plot_augmentation(datagenerator, image_i, name):
  '''
  Plot augmentation defined in a Keras ImageDataGenerator.
  Plots an image from "Warp-C/train/cardboard" in 16 versions. 
  Saves the plot in the 'out' folder.

  Arguments:
  - datagenerator: A Keras ImageDataGenerator
  - image_i: the index of what image to plot
  - name: prefix name to give to the saved plot.

  Returns:
    None

  Source:
    Code to plot the augmented images have been taken from this website: https://gac6.medium.com/visualizing-data-augmentations-from-keras-image-data-generator-44f040aa4c9f

  '''
  # define path to train-cardboard directory
  dir = os.path.join("Warp-C", "train", "cardboard")  
  images = sorted(os.listdir(dir))

  # load chosen image
  img = mpimg.imread(os.path.join(dir, images[image_i]))
  plt.imshow(img)

  # save the original image with no augmentations or preprocessing
  plt.savefig(os.path.join("out", 'original_image.png'))

  # convert image to tensors
  img_tensor = img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)

  # apply augmentation to the single image
  pic = datagenerator.flow(img_tensor, batch_size =1)
  plt.figure(figsize=(16, 16))
 
  # plot 16 figures
  for i in range(1,17):
    plt.subplot(4, 4, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_)
    plt.show()
    plt.savefig(os.path.join("out", f'{name}_augmentation_example.png'))
