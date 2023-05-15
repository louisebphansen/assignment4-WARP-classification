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

def create_report(model, test_gen, filename):
    
    pred = model.predict_generator(test_gen)

    predicted_classes = np.argmax(pred,axis=1)

    test_gen.reset()

    y_true = test_gen.classes

    labels = ['cardboard', 'glass', 'metal', 'plastic']

    report = classification_report(y_true,
                            predicted_classes, target_names = labels)
    
    out_path = os.path.join("out", "unbalanced", f"{filename}_classification_report.txt")

    with open(out_path, 'w') as file:
                file.write(report)


def run_model(train_gen, val_gen, test_gen, filename):
    model = build_model()

    model.fit_generator(generator=train_gen, # fit model with generators
                        steps_per_epoch=128,
                        validation_data=val_gen,
                        validation_steps=128,
                        epochs=1)
    
    create_report(model, test_gen, filename)

def main():
    run_model(dt.train_gen_none, dt.val_gen_none, dt.test_gen_none, 'no_aug')
    run_model(dt.train_gen_low, dt.val_gen_low, dt.test_gen_low, 'low_aug')
    run_model(dt.train_gen_high, dt.val_gen_high, dt.test_gen_high, 'high_aug')

if __name__ == '__main__':
   main()