'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, FINAL PROJECT: Waste classification using CNN's

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:


'''


import os
import pandas as pd
import argparse
import numpy as np

# tf tools
import tensorflow as tf

from utils import plot_augmentation
# layers
# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)

# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input)

def define_generators(subset, augmentation_level):
    '''
    Create Keras ImageDataGenerator based on data subset and augmentation level.
    Saves a plot in the 'out' folder with example of applying augmentation and the original image

    Arguments:
        - subset: what subset of the data to use. Either "train" or "test"
        - augmentation_level: level of augmentation. Either "none", "low" or "high"
        
    Returns:
        - Keras ImageDataGenerator 
    '''

    # test data should not be augmented, just preprossed like the training
    if subset == 'test' and augmentation_level == 'none':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input)

    # define train generator with no augmentation, just preprocessing
    elif subset == 'train' and augmentation_level == 'none':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            validation_split = 0.2) # split training data to 80% train, 20% val

        # plot example of the augmentation using the first image in the "train/cardboard" folder
        plot_augmentation(datagenerator, 0, 'no_aug')
    
    # define train generator with low augmentation
    elif subset == 'train' and augmentation_level == 'low':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            horizontal_flip=True, 
            vertical_flip=True,
            validation_split = 0.2)

        plot_augmentation(datagenerator, 0, 'low_aug')
    
    # define train generator with high augmentation
    elif subset == 'train' and augmentation_level == 'high':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            horizontal_flip=True, 
            vertical_flip=True,
            rotation_range=20,
            zoom_range = [0.9, 1.25],
            brightness_range = (0.7, 1.2),
            validation_split = 0.2)

        plot_augmentation(datagenerator, 0, 'high_aug')

    else: 
        raise ValueError('Not a possible argument')

    return datagenerator

def create_flow(datagenerator, subset):
    '''
    Creates batches of data from the 'Warp-C' directory using Kera's 'flow_from_directory'.

    Arguments:
        - datagenerator: A Keras imagedatagenerator 
        - subset: subset of the data, can be 'train', 'val' or 'test' 

    Returns:
        - A Keras DataIterator that can be used for fitting a model
    '''

    if subset == 'train':
        gen = datagenerator.flow_from_directory(
                directory= os.path.join("Warp-C", "train"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = True,
                seed = 2830,
                subset = 'training')

    elif subset == 'val':
        gen = datagenerator.flow_from_directory(
                directory= os.path.join("Warp-C", "train"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = True, 
                seed = 2830,
                subset = 'validation')
        
    elif subset == 'test':
        gen = datagenerator.flow_from_directory(        
                directory= os.path.join("Warp-C", "test"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = False,
                seed = 2830)
    
    else:
        raise ValueError('Not a possible argument')

    return gen


def prep_data(augmentation_level):
    '''
    Create ImageDataGenerators and generate batches of data to be used in model fitting

    Arguments:
        - augmentation_level: level of data augmentation. Must be 'none', 'low' or 'high'.

    Returns:
        - Keras DataIterators for train, validation and test data
    '''

    test_image_gen = define_generators('test', 'none')
    train_image_gen = define_generators('train', augmentation_level)

    test_gen = create_flow(test_image_gen, 'test')
    train_gen = create_flow(train_image_gen, 'train')
    val_gen = create_flow(train_image_gen, 'val')

    return train_gen, val_gen, test_gen
