# Final Project: Waste Classification using CNN's

This assignment is the final project on the Visual Analytics course on the elective in Cultural Data Science at Aarhus University. 

### Contributions

The code was created by me. However, code provided throughout the course has been reused and adapted for this project. Furthermore, some code from [WEBSITE] has been reused. 

### Repository description
This project aims to apply the tools and methods acquired in the Visual Analytics course to create a classifier which can distinguish between four different types of waste. The repository contains code to run a classifier using different levels and methods of augmentation as well as balanced vs unbalanced data. The motivation behind the project was due to an interest in applying image classification and computer vision to a real life problem such as waste recyling.


### Methods and contents

#### Contents
| Folder/File  | Contents| Description |
| :---:   | :---: | :--- |
| ```out```     |```balanced``` ```imbalanced``` | Contains the output of running the **classify.py** script. Main folder contains examples of augmented images at different levels, where the two subfolders contain the history plots and classification reports for running the model with different levels of augmentation. |
| ```src```   | **classify.py** **data.py** **utils.py**| Contains the Python scripts to run a classifier on the WaRP dataset. **data.py** contains code to load, prepare and augment the data (using Keras ImageDataGenerator). **classify.py** contains code to train a classifier on the WaRP data. **utils.py** contains functions to plot model history and examples of augmented images.|
|README.md| README.md | Description of the repository and how to use it. |
|requirements.txt| requirements.txt| Packages required to run the code.|
|run.sh| run.sh| Bash script for running the **classify.py** script, with already defined arguments.|
|setup.sh|setup.sh|Bash script for setting up virtual environment for project|

#### Data

##### Description of data
The WaRP data was found on Kaggle (https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset) The Kaggle page contains data for image segmentation, detection or classification. The classification data was used for this project. The data consists of *8823* color images for training and *1523* color images for testing. The original data contains 28 different classes of waste, with 17 different categories of plastic bottles, three different types of glass bottles, four different types of detergent, two different types of cardboard as well as cans and canisters. I thought it would be more useful in a real-life scenario to use more general and broad classes for the waste. Thus, I decided to create four new classes by merging the 28 original subclasses. The contents of the classes are as follows:

| New class  | Previous subclasses| Number of merged subclasses|
| :---:   | :---: |  :---: |
|Metal| *cans* | 1|
|Cardboard|*juice-cardboard*, *milk-cardboard*|2 | 
|Glass|*glass-dark*, *glass-green*, *glass-transp*| 3|
|Plastic|*bottle-blue*, *bottle-blue-full*, *bottle-blue5l*, *bottle-blue5l-full*, *bottle-dark*, *bottle-dark-full*, *bottle-green*, *bottle-green-full*, *bottle-milk*, *bottle-milk-full*, *bottle-multicolor*, *bottle-multicolorv-full*, *bottle-oil*, *bottle-oil-full, bottle-transp, bottle-transp-full, bottle-yogurt, detergent-box, detergent-color, detergent-transparent, detergent-white, canister*| 22|


##### Acquire the data
To reproduce the results in this repository, download the data here (https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset) and unzip the file, which will be called 'archive'. Inside the unzipped archive folder, you will find several subfolders. Choose the one called 'Warp-C' and place it in the ```assignment4-WARP-classification``` folder (i.e., the main folder/repository). In order to create the desired four classes as explained above, some of the folders needs to be merged. To do so, run the ```move.sh``` script in the terminal **from the Warp-C folder**. I.e., your code should look like this:

```
assignment4-WARP-classification/Warp-C bash move.sh
```

This will create two new folders called ```train``` and ```test``` inside the ```Warp-C``` folder, which each contains the subfolders (i.e., the four new classes), ```cardboard```, ```glass```, ```metal``` and ```plastic``` with the images from the old subfolders as described above. This is the data that will be used for training the classifier. 

##### Examples of training data


Your repository should look like this:

**assignment4-WaRP-classification (MAIN FOLDER)**
out
src
README.md
requirements.txt
run.sh
setup.sh
Warp-C
    --> train
        --> cardboard
        --> glass
        --> metal
        --> plastic
    --> test
        --> cardboard
        --> glass
        --> metal
        --> plastic
    --> train_crops
    --> test_crops
    --> move.sh

##### A note on the data
There is a notable difference in the amount of samples in each of the categories. In the training data, **plastic** contains ___ samples, **write number of samples for train and test data!**. This could have a big influence on the training data, as the model would be prone to overfit on the classes with more samples. To account and test for this imbalance, the model can be run in a script that balances the data (by calculating class weights, thereby letting the prominent classes weigh less in the model) and one that does not. See **Usage** and **Results** for more on this issue.

#### Methods

*The following section describes the methods used in the provided Python scripts.*

**DataGenerators and augmentation**
The code uses Kera's ImageDataGenerators to generate batches of data to feed the model. This relies less on your computer's memory, which makes it easier to run, and allows for real-time data augmentation when the model is fit. To prepare data to train the model, the training data is split into training and validation data with a 80/20 split. A seperate ImageDataGenerator is created for the training, validation and testing datasets. Furthermore, three different levels of augmentation can be applied to the data (see **Usage/Arguments** on how to run the code). 

**When not applying any augmentation, the image is only preprocessed by converting the pixels from RGB to BGR and each color channel is zero-centered.** what? 

**indsæt billede**

For the 'low' level of augmentation, the same preprocessing as described above is applied, and horizontal and vertical flip is set to *True*, allowing for a simple augmentation.

**indsæt billede**

For the 'high' level of augmentation, several augmentation methods are defined. The images are preprocessed like the other levels, and horizontal and vertical flip is also set to true. Furthermore, a zoom and rotation range is defined as well as a brightness range.
**indsæt billede**

**Classification**
A classifier is trained on the train, validation and test generators. The classifier uses a pretrained model, VGG16, without its classification layers. Instead, two new classification layers are added, of size 256 and 128. The model uses ReLU as its activation function, stochasitc gradient descent as the optimizer, and categorical crossentropy as the loss function. The output layer consists of the four classes representing the four different types of waste.


### Usage
All code for this project was designed to run on an Ubuntu ** operating system. 

To reproduce the results in this repository, clone this repository using ```git clone```. To sort the data into the correct folders, run the ```move.sh``` script as explained in the **Data** section.

**Setup**
First, ensure that you have installed the **venv** package for Python(if not, run ```sudo apt-get update``` and ```sudo apt-get install python3-venv```). To set up the virtual environment, run ```bash setup.sh``` from the terminal, from the main folder.

**Run code**
To run the code, you can do the following:

##### Run script(s) with predefined arguments

From the terminal (and still in the main folder) type ```bash run.sh``` which activates the virtual environment and runs *six* classifier scripts, three scripts with different augmentation levels (none, low, medium) using balanced data, and three scripts with same three augmentations levels using imbalanced data. All six models are run with 10 epochs. The results from this run can be seen in the ```out``` folder and is described in the **Results** section.

**NB:** The code may take several hours to run, depending on the processing power of your machine. 

##### Define arguments yourself

From the terminal, activate the virtual environment and run the script with the desired arguments:

```
# activate virtual environment
source env/bin/activate

# run script
python3 src/classify.py --epochs <epochs> --balance <balance> --augmentation_level <augmentation_level> 
```

**Arguments:**
- **epochs:** number of epochs to run the model for
- **balance**: whether to balance the data by using weighted classes or run the model the original data. Must be either 'balanced' for using balanced data or 'imbalanced' to use imbalanced data.
- **augmentation_level**: Level of augmentation to apply to the data. Must be 'none', 'low', or 'high'.

### Results