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

    if subset == 'train' and augmentation_level == 'none':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            validation_split = 0.2)
    
    if subset == 'train' and augmentation_level == 'low':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            horizontal_flip=True, 
            rotation_range=15,
            validation_split = 0.2)
    
    if subset == 'train' and augmentation_level == 'high':
        datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            horizontal_flip=True, 
            rotation_range=20,
            zoom_range = [0, 1.25],
            width_shift_range=0.2,
            height_shift_range=0.2,
            validation_split = 0.2,
            brightness_range = (1, 2))

    # add ELSE STATEMENT!

    return datagenerator
    

def build_model():
    '''
    Build a convolutional neural network using the pretrained VGG16 model as feature extractor. Model has two classification layers and a final output layer.
    Code is adapted from the Session 9 notebook of the Visual Analytics course at AU, 2023.
    
    Returns:
    A compiled model that can be fit and used for a classification task.
    
    '''
    
    # load model without classifier layers
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(224, 224, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    bn = BatchNormalization()(flat1)
    class1 = Dense(256, 
                activation='relu')(bn)
    class2 = Dense(128, 
                activation='relu')(class1)
    output = Dense(4, 
                activation='softmax')(class2)

    # define new model
    model = Model(inputs=model.inputs, 
              outputs=output)

    # compile
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


def create_generators(datagenerator, subset):
    
    if subset == 'train':
        gen = datagenerator.flow_from_directory(
                directory= os.path.join("Warp-C", "train"),
                target_size = (224, 224),
                color_mode = 'rgb',
                classes = ['cardboard', 'glass', 'metal', 'plastic'],
                shuffle = True, 
                save_to_dir = 'augmented_images',
                save_prefix = 'augmented',
                subset = 'training')

    if subset == 'val':
        gen = datagenerator.flow_from_directory(
                directory= os.path.join("Warp-C", "train"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = True, 
                save_to_dir = 'augmented_images',
                save_prefix = 'augmented',
                subset = 'validation')
        
    if subset == 'test':
        gen = datagenerator.flow_from_directory(        
                directory= os.path.join("Warp-C", "test"),
                target_size = (224, 224),
                color_mode = 'rgb',
                shuffle = False)

    return gen


def create_class_weights(traingenerator):

    traingenerator.reset()

    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(traingenerator.classes),
                                                 y = traingenerator.classes)

    class_weights = dict(zip(np.unique(traingenerator.classes), class_weights))

    return class_weights


def main():

    test_image_gen = define_generators('test', 'none')
    train_image_gen = define_generators('train', 'high')

    test_gen = create_generators(test_image_gen, 'test')
    train_gen = create_generators(train_image_gen, 'train')
    val_gen = create_generators(train_image_gen, 'val')

    class_weights = create_class_weights(train_gen)

    model = build_model()

    model.fit_generator(generator=train_gen, # fit model with generators
                        steps_per_epoch=128,
                        validation_data=val_gen,
                        validation_steps=128,
                        epochs=8, 
                        class_weight = class_weights)

    pred = model.predict_generator(test_gen)
    predicted_classes = np.argmax(pred,axis=1)

    test_gen.reset()

    y_true = test_gen.classes

    labels = ['cardboard', 'glass', 'metal', 'plastic']

    report = classification_report(y_true,
                            predicted_classes, target_names = labels)
    
    out_path = os.path.join("out", f"high_aug_classification_report.txt")

    with open(out_path, 'w') as file:
                file.write(report)


if __name__ == '__main__':
   main()