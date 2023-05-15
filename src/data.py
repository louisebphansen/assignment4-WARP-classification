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
from tensorflow.keras.applications.vgg16 import (preprocess_input, VGG16)

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

def define_generators(subset, augmentation_level):
    
    if subset == 'test' and augmentation_level == 'none':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input)

    elif subset == 'train' and augmentation_level == 'none':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            validation_split = 0.2)
    
    elif subset == 'train' and augmentation_level == 'low':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            horizontal_flip=True, 
            vertical_flip=True,
            validation_split = 0.2)
    
    elif subset == 'train' and augmentation_level == 'high':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            horizontal_flip=True, 
            vertical_flip=True,
            rotation_range=120,
            zoom_range = [0.2, 1.25],
            validation_split = 0.2,
            brightness_range = (0.2, 1.2))

    else:
        raise ValueError('Not a possible argument')

    return datagenerator

def create_generators(datagenerator, subset):
    
    if subset == 'train':
        gen = datagenerator.flow_from_directory(
                directory= os.path.join("Warp-C", "train"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = True, 
                save_to_dir = 'augmented_images',
                save_prefix = 'augmented',
                subset = 'training')

    elif subset == 'val':
        gen = datagenerator.flow_from_directory(
                directory= os.path.join("Warp-C", "train"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = True, 
                subset = 'validation')
        
    elif subset == 'test':
        gen = datagenerator.flow_from_directory(        
                directory= os.path.join("Warp-C", "test"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = False)
    
    else:
        raise ValueError('Not a possible argument')

    return gen


def prep_data(augmentation_level):

    test_image_gen = define_generators('test', 'none')
    train_image_gen = define_generators('train', augmentation_level)

    test_gen = create_generators(test_image_gen, 'test')
    train_gen = create_generators(train_image_gen, 'train')
    val_gen = create_generators(train_image_gen, 'val')

    return train_gen, val_gen, test_gen


#train_gen_none, test_gen_none, val_gen_none = prep_data('none')
#train_gen_low, test_gen_low, val_gen_low = prep_data('low')
#train_gen_high, test_gen_high, val_gen_high = prep_data('high')