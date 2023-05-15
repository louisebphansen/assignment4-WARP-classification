# Final Project: Waste Classification using CNN's

This assignment is the final project on the Visual Analytics course on the elective in Cultural Data Science at Aarhus University. 

### Contributions

The code was created by me. However, code provided throughout the course has been reused and adapted for this project. Furthermore, some code from [WEBSITE] has been reused. 

### Project description

This project aims to apply the tools and methods taught in the course in a waste-recycling scenario. More specifically, a dataset from Kaggle, 'WaRP' (Waste Recycling Plant) has been used. I thought it would be interesting to apply image classification and computer vision tools to a real life problem to see if it could be a possible method for optimizing waste recycling. **skriv mere her, det er dårligt** .. 

### Contents

LAV PÆN OVERSIGT OVER REPO!!


### Methods and contents

#### Data

##### Description of data
The WaRP data was found on Kaggle, **insert link** The image classification data was used for this assignment. The data consists of 8823 images for training and **test_images** for testing. The original data contains 28 different classes, with 17 different categories of plastic bottles, three different types of glass bottles, four different types of detergent and two different types of cardboards. I thought it would be more useful in a real-life situation to use more *overall** classes for the waste. Thus, I decided to create 4 classes consisting of the 28 original subclasses. The contents of the classes are as follows:

**Cardboard**
- juice-cardboard
- milk-cardboard

**Glass**
- glass-dark
- glass-green
- glass-transp

**Metal**
- Cans

**Plastic**
- All 17 categories of plastic bottles (subfolders starting with *bottle*)
- All 4 detergent containers
- Canisters

##### Acquire the data
To reproduce the results in this repository, download the data here and unzip the file, which will be called 'archive'. Inside the zipped file, you will find several subfolders. Choose the one called 'Warp-C' and place it in the ```assignment4-WARP-classification``` folder. In order to create the desired four classes as explained above, some of the folders needs to be merged. To do so, run the ```move.sh``` script in the terminal **from the Warp-C folder**. I.e., your code should look like this:

```
assignment4-WARP-classification/Warp-C bash move.sh

```

This will create two new folders called ```train``` and ```test```, which each contains the subfolders (i.e., the four new classes), ```cardboard```, ```glass```, ```metal``` and ```plastic```.

SKRIVE HVOR MANGE DER ER I HVER SUBFOLDER

"YOUR REPO SHOULD LOOK LIKE THIS".

This is the data that will be used for training the classifier. 

### LAV EN BESKRIVELSE OVER HVORDAN DATAEN SER UD !!!!

```data.py``` contains code to construct ImageDataGenerators using the data from the ```train``` and ```test``` folder with different levels of data augmentation. 

```classification.py``` contains code to train a Convolutional Neural Network which uses a pretrained model (VGG16) as a feature extractor and has two classification layers of size **SIZE** and **SIZE**. The results of the classification problem is saved in the ```out```folder. Due to the imbalanced nature of the data (see **Data** section), the script can be run with code that balances the data or code that does not. 

