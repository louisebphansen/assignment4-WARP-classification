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

  dir = os.path.join("Warp-C", "train", "cardboard")  
  images = sorted(os.listdir(dir))
  img = mpimg.imread(os.path.join(dir, images[image_i]))
  plt.imshow(img)
  plt.savefig(os.path.join("out", 'original_image.png'))

  img_tensor = img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)

  pic = datagenerator.flow(img_tensor, batch_size =1)
  plt.figure(figsize=(16, 16))
  #Plots our figures

  for i in range(1,17):
    plt.subplot(4, 4, i)
    batch = pic.next()
    image_ = batch[0].astype('uint8')
    plt.imshow(image_)
    plt.show()
    plt.savefig(os.path.join("out", f'{name}_augmentation_example.png'))
