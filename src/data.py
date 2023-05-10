import os
import pandas as pd
import argparse
import numpy as np

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input)


datagenerator = ImageDataGenerator(
        preprocessing_function = preprocess_input) 

    # fetch data from directory
    gen = datagenerator.flow_from_directory(
        dataframe=df,
        directory=directory,
        x_col='image_path',
        y_col='class_label',
        batch_size=32,
        shuffle=True,
        class_mode="categorical",
        target_size=(size,size))
    
    return gen