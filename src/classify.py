'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, FINAL PROJECT: Waste classification using CNN's

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
The script trains a classifier on batches of data from an altered version of the WaRP dataset, classifying four different types of waste. 
When running the script, it is possible to specify whether to balance the unequal number of samples, and what level of augmentation to apply to the data.
The script saves a classification report and a history plot in the 'out' folder, as well as examples of augmented images.

'''

# generic tools
import os
import argparse
import numpy as np

# tf tools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)

# tf model tools
from tensorflow.keras.applications.vgg16 import (preprocess_input, VGG16)

from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

# define arguments that can be defined by the user from command line
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help = "How many epochs the model should run for", default = 10)
    parser.add_argument("--balance", help = "Whether to balance the unequal number of samples in each class or not", default = 'imbalanced')
    parser.add_argument("--augmentation_level", help = "Level of augmentation to apply to the data. Must be none, low or high", default = 'none')
    args = vars(parser.parse_args())
    
    return args

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
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)

    sgd = SGD(learning_rate=lr_schedule)

    model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model

def create_class_weights(traingenerator):
    '''
    Create class weights based on number of samples in each class.

    Arguments:
        - traingenerator: Keras DirectoryIterator with training data

    Returns:
        - a dictionary of class weights for each of the classes

    Sources:
        Code to compute class weights was found here:
        https://stackoverflow.com/questions/69783897/compute-class-weight-function-issue-in-sklearn-library-when-used-in-keras-cl 

    '''

    # compute class weights based on number of samples in each class
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(traingenerator.classes),
                                                 y = traingenerator.classes)

    # convert to a dictionary suitable for using in a model.fit pipeline
    class_weights = dict(zip(np.unique(traingenerator.classes), class_weights))

    return class_weights

def create_report(model, test_gen, filename):
    '''
    Predicts test data from model and saves the classification report.
    
    Arguments:
    - Model: a trained model
    - test_gen: Keras DirectoryIterator with test data
    - filename: what to call the prefix of the classification report

    Returns:
        None
    '''

    # predict test data using model
    pred = model.predict(test_gen)
    predicted_classes = np.argmax(pred,axis=1)

    # reset test generator to start from first batch (to match outputs to predicted data)
    test_gen.reset()

    # get true labels of test data
    y_true = test_gen.classes

    labels = list(test_gen.class_indices.keys())

    # create classification report from predicted and true labels
    report = classification_report(y_true,
                            predicted_classes, target_names = labels)
    
    # save report
    out_path = os.path.join("out", f"{filename}_TEST_classification_report.txt")

    with open(out_path, 'w') as file:
                file.write(report)


def run_model_balanced(train_gen, val_gen, test_gen, num_epochs, filename):
    ''' 
    Fits a model using training and validation data and predicts on test data.
    Uses class weights to account for unequal number of samples in training data.
    Saves a plot of model fitting history and a classification report.

    Arguments:
        - train_gen: Keras DirectoryIterator with batches of train data
        - val_gen: Keras DirectoryIterator with batches of validation data
        - test_gen: Keras DirectoryIterator with batches of test data
        - num_epochs: Number of epochs to fit the model for
        - filename: prefix to what the plot and classification report files should be named
    
    Returns:
        None

    '''
    # compile model
    model = build_model()

    # create class weights based on number of samples in each class
    class_weights = create_class_weights(train_gen)

    # fit model with generators and set class weights
    H = model.fit(train_gen,
                    steps_per_epoch=len(train_gen), # number of batches in train data
                    validation_data=val_gen,
                    validation_steps=len(val_gen), # number of batches in val data 
                    class_weight = class_weights,
                    epochs=num_epochs)
    
    from utils import save_plot_history

    # save history plot
    save_plot_history(H, num_epochs, filename)

    # save classification report
    create_report(model, test_gen, filename)

def run_model_imbalanced(train_gen, val_gen, test_gen, num_epochs, filename):
    ''' 
    Fits a model using training and validation data and predicts on test data.
    Does not account for unequal number of samples in each class.
    Saves a plot of model fitting history and a classification report.

    Arguments:
        - train_gen: Keras DirectoryIterator with batches of train data
        - val_gen: Keras DirectoryIterator with batches of validation data
        - test_gen: Keras DirectoryIterator with batches of test data
        - num_epochs: Number of epochs to fit the model for
        - filename: prefix to where the plot and classification report files should be named
    
    Returns:
        None

    '''    
    # compile model
    model = build_model()

    # fit model with generators and no class weights
    H = model.fit(train_gen, 
                        steps_per_epoch=len(train_gen), # number of batches in train data
                        validation_data=val_gen,
                        validation_steps=len(val_gen), # number of batches in validation data
                        epochs=num_epochs)
    
    from utils import save_plot_history

    # save history plot
    save_plot_history(H, num_epochs, filename)
    
    # save classification report
    create_report(model, test_gen, filename)


def main():
    
    # parse arguments
    args = argument_parser()

    # import functions from data script which prepares the data
    from data import prep_data

    # create batches of data based on chosen level of data augmentation
    train_gen, val_gen, test_gen = prep_data(args['augmentation_level'])

    # run the model with balanced or imbalanced data
    if args['balance'] == 'balanced':
        run_model_balanced(train_gen, val_gen, test_gen, args['epochs'], f"balanced/{args['augmentation_level']}")

    elif args['balance'] == 'imbalanced':
        run_model_imbalanced(train_gen, val_gen, test_gen, args['epochs'], f"imbalanced/{args['augmentation_level']}")

if __name__ == '__main__':
   main()