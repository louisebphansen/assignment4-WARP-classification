import os
import argparse
import numpy as np

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)

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

import data as dt

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help = "how many epochs the model should run for")
    #parser.add_argument("--classes", type=int, help= "number of classes in the data")
    #parser.add_argument("--out_folder", help="where to save the output classification report")
    parser.add_argument("--balanced", type=bool, help = "whether to balance the unequal number of samples in each class or not")
    parser.add_argument("--augmentation_level", help = "must be none, low or high")
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
        initial_learning_rate=0.01,
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
        - traingenerator: ImageDataGenerator for training data

    Returns:
        - a dictionary of class weights for each of the classes
    '''

    # compute class weights based on number of samples in each class
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                 classes = np.unique(traingenerator.classes),
                                                 y = traingenerator.classes)

    # convert to a dictionary suitable for using in a fit_generator pipeline
    class_weights = dict(zip(np.unique(traingenerator.classes), class_weights))

    return class_weights

def create_report(model, test_gen, filename):
    '''
    Predicts test data from model and saves the classification report.
    
    Arguments:
    - Model: a trained model
    - test_gen: Keras ImageDataGenerator with test data (i.e., shuffle must be False)
    - filename: prefix of the classification report

    Returns:
        None
    '''

    # predict test data using model
    pred = model.predict_generator(test_gen)
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
    out_path = os.path.join("out", f"{filename}_classification_report.txt")

    with open(out_path, 'w') as file:
                file.write(report)


def run_model_balanced(train_gen, val_gen, test_gen, epochs, filename):
    
    model = build_model()

    class_weights = create_class_weights(train_gen)

    model.fit_generator(generator=train_gen, # fit model with generators
                        steps_per_epoch=128,
                        validation_data=val_gen,
                        validation_steps=128,
                        epochs=epochs, 
                        class_weight = class_weights)
    
    create_report(model, test_gen, filename)

def run_model_imbalanced(train_gen, val_gen, test_gen, epochs, filename):
    
    model = build_model()

    H = model.fit_generator(generator=train_gen, # fit model with generators
                        steps_per_epoch=128,
                        validation_data=val_gen,
                        validation_steps=128,
                        epochs=epochs)
    
    from utils import save_plot_history
    save_plot_history(H, epochs, filename)
    
    create_report(model, test_gen, filename)

def mainlol():

    args = argument_parser()

    if args['balanced'] == True:
        run_model_balanced(dt.train_gen_none, dt.val_gen_none, dt.test_gen_none, args['epochs'], 'balanced/no_aug')
        run_model_balanced(dt.train_gen_low, dt.val_gen_low, dt.test_gen_low, args['epochs'], 'balanced/low_aug')
        run_model_balanced(dt.train_gen_high, dt.val_gen_high, dt.test_gen_high, args['epochs'], 'balanced/high_aug')

    if args['balanced'] == False:
        run_model_imbalanced(dt.train_gen_none, dt.val_gen_none, dt.test_gen_none, args['epochs'], 'imbalanced/no_aug')
        run_model_imbalanced(dt.train_gen_low, dt.val_gen_low, dt.test_gen_low, args['epochs'], 'imbalanced/low_aug')
        run_model_imbalanced(dt.train_gen_high, dt.val_gen_high, dt.test_gen_high, args['epochs'], 'imbalanced/high_aug')

def main():
    
    args = argument_parser()

    # ellers kunne man også bare importere funktionen stadig, men definere de forskellige variabler? ved ikke om det er bedre..  
    from data import prep_data
    train_gen, val_gen, test_gen = prep_data(args['augmentation_level'])

    if args['balanced'] == True:
        run_model_balanced(train_gen, val_gen, test_gen, args['epochs'], f"balanced/{args['augmentation_level']}")

    if args['balanced'] == False:
        run_model_imbalanced(train_gen, val_gen, test_gen, args['epochs'], f"imbalanced/{args['augmentation_level']}")

if __name__ == '__main__':
   main()