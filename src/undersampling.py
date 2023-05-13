import os
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


files = os.listdir(os.path.join("Warp-C", "train", "cardboard"))

cardboard = (len(os.listdir(os.path.join("Warp-C", "train", "cardboard"))))
glass = (len(os.listdir(os.path.join("Warp-C", "train", "glass"))))
metal = (len(os.listdir(os.path.join("Warp-C", "train", "metal"))))
plastic = (len(os.listdir(os.path.join("Warp-C", "train", "plastic"))))

print(cardboard)
print(glass)
print(metal)
print(plastic)

datagenerator = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        validation_split = 0.2)

train_gen = datagenerator.flow_from_directory(
            directory= os.path.join("Warp-C", "train"),
            target_size = (224, 224),
            color_mode = 'rgb',
            classes = ['cardboard', 'glass', 'metal', 'plastic'],
            shuffle = True, 
            save_to_dir = 'augmented_images',
            save_prefix = 'augmented',
            subset = 'training')


from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
model.fit(X_train, y_train, class_weight=class_weights)