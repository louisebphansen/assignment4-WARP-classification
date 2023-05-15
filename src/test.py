
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
from tensorflow.keras.applications.vgg16 import (preprocess_input)

import numpy as np
import os

datagenerator = ImageDataGenerator(
            preprocessing_function = preprocess_input,
            horizontal_flip=True, 
            vertical_flip=True,
            rotation_range=120,
            zoom_range = [0.2, 1.25],
            validation_split = 0.2,
            brightness_range = (0.2, 1.2))

def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)

img = mpimg.imread(os.path.join("Warp-C", "train", "cardboard", "train_crops_cardboard_juice-cardboard_Monitoring_photo_04-Mar_04-26-41_01.jpg"))
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
#plt.savefig('plot_test.png')