# Final Project: Waste Classification using CNN's

This assignment is the final project on the Visual Analytics course on the elective in Cultural Data Science at Aarhus University. 

### Contributions

The code was created by me. However, code provided throughout the course has been reused and adapted for this project. 

### Repository description
This project aims to apply the tools and methods acquired in the Visual Analytics course to create a classifier which can distinguish between four different types of waste. The repository contains code to run a classifier using different levels and methods of augmentation as well as balanced vs imbalanced data. The motivation behind the project was due to an interest in applying image classification and computer vision to a real life problem such as waste recyling.

### Contents of the repository
| Folder/File  | Contents| Description |
| :---:   | :---: | :--- |
| ```out```     |```balanced```, ```imbalanced```, *high_aug_augmentation_example.png*, *low_aug_augmentation_example.png*, *no_aug_augmentation_example.png*, *original_image.png*  | Contains the output of running the **classify.py** script. The main folder contains examples of augmented images at different levels as well as the original image with no augmentation or preprocessing. The two subfolders contain the history plots and classification reports for running the model with different levels of augmentation with either balanced or imbalanced data. |
| ```src```   | **classify.py**, **data.py**, **utils.py**| Contains the Python scripts to run a classifier on the WaRP dataset. **data.py** contains code to load, prepare and augment the data (using Keras ImageDataGenerator). **classify.py** contains code to train a classifier. **utils.py** contains functions to plot model history and examples of augmented images.|
|README.md| - | Description of the repository and how to use it. |
|move_files.sh| - |Bash script for creating new train and test folders with subdirectories. |
|requirements.txt| - | Packages required to run the code.|
|run.sh| - | Bash script for running the **classify.py** script, with already defined arguments.|
|setup.sh| - |Bash script for setting up virtual environment for project|

### Data
The WaRP data was found on Kaggle (https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset) The Kaggle page contains data for image segmentation, detection or classification. The classification data was used for this project. The data consists of *8823* color images for training and *1521* color images for testing. The original data contains 28 different classes of waste, with 17 different categories of plastic bottles, three different types of glass bottles, four different types of detergent, two different types of cardboard as well as cans and canisters. I thought it would be more useful in a real-life scenario to use more general and broad classes for the waste. Thus, I decided to create four new classes by merging the 28 original subclasses. The contents of the classes are as follows:

| New class  | Previous subclasses| Number of merged subclasses|
| :---:   | :---: |  :---: |
|Metal| *cans* | 1|
|Cardboard|*juice-cardboard*, *milk-cardboard*|2 | 
|Glass|*glass-dark*, *glass-green*, *glass-transp*| 3|
|Plastic|*bottle-blue*, *bottle-blue-full*, *bottle-blue5l*, *bottle-blue5l-full*, *bottle-dark*, *bottle-dark-full*, *bottle-green*, *bottle-green-full*, *bottle-milk*, *bottle-milk-full*, *bottle-multicolor*, *bottle-multicolorv-full*, *bottle-oil*, *bottle-oil-full, bottle-transp, bottle-transp-full, bottle-yogurt, detergent-box, detergent-color, detergent-transparent, detergent-white, canister*| 22|

#### Examples of training data

Below is an example of the four different classes of waste used for this project.

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/c33e8192-9dfe-42bf-bad3-94e1355be746" width="600">

#### Data overview

| Subset  | Class| Samples|
| :---:   | :---: |  :---: |
|Train|Cardboard| 650 |
|Train|Glass| 448 |
|Train|Metal| 562 |
|Train|Plastic| 7163 |
|Test|Cardboard| 162 |
|Test|Glass| 86 |
|Test|Metal| 98 |
|Test|Plastic| 1175 |

As seen from the above table, there is a notable difference in the amount of samples in each of the categories. This could have a big influence on training the model, as it would be prone to overfit on the classes with more samples and not learn very much about the classes with few samples. To account and test for the effect of this imbalance, the model can be run in a script that balances the data (by calculating class weights, thereby letting the prominent classes weigh less in the model) and one that does not. See **Usage** and **Results** for more on this issue.

### Methods

*The following section describes the methods used in the provided Python scripts.*

#### DataGenerators and augmentation

The code uses Kera's ImageDataGenerators to generate batches of data to feed the model. This relies less on your computer's memory, which makes it easier to run, and allows for real-time data augmentation when the model is fit. To prepare data to train the model, the training data is split into training and validation data with a 80/20 split. A seperate ImageDataGenerator is created for the training, validation and testing datasets. Furthermore, three different levels of augmentation can be applied to the data (see **Usage/Arguments** on how to run the code). 

##### No augmentation

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/a29c61f7-cd7e-4ed9-836d-0ae1e6d84c21" width="650">

When not applying any augmentation, the image is only preprocessed by converting the pixels from RGB (*red-green-blue*) to BGR and zero-centering color channels. 

##### Low augmentation

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/991605f4-778a-42a2-9e19-874062c18612" width="650">

For the 'low' level of augmentation, the same preprocessing as described above is applied, and horizontal and vertical flip is set to *True*, allowing for a simple augmentation.

##### High augmentation

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/f72912c4-6cb1-49aa-9a0d-adb0caa003d2" width="650">

For the 'high' level of augmentation, several augmentation methods are defined. The images are preprocessed like the other levels, and horizontal and vertical flip is also set to true. Furthermore, a zoom and rotation range is defined as well as a brightness range.


#### Classification

A convolutional neural network (CNN) classifier is trained on the train, validation and test generators. The classifier uses a pretrained CNN, VGG16, without its classification layers. Instead, two new classification layers are added, of size 256 and 128. The model uses ReLU as its activation function, stochastic gradient descent as the optimizer, and categorical crossentropy as the loss function. The output layer consists of the four classes representing the four different types of waste.


### Usage
All code for this project was designed to run on an *Ubuntu 22.10* operating system. 

To reproduce the results in this repository, clone it using ```git clone```. 

It is important that you run all scripts from the main folder, i.e., *assignment4-WARP-classification*. Your terminal should look like this:

```
--your_path-- % assignment4-WARP-classification % 
```

#### Acquire the data
The data is not attached to this repository because of its size. Instead, download the data here: https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset

Unzip the file, which will be called 'archive'. Inside the unzipped archive folder, you will find several subfolders. Choose the one called 'Warp-C' and place it in the ```assignment4-WARP-classification``` folder (i.e., the main folder/repository). 

In order to create the desired four classes as explained above, the folders needs to be merged into new classes. To do so, run the ```move_files.sh``` script from the terminal by typing ```bash move_files.sh```. 

This will create two new folders called ```train``` and ```test``` inside the ```Warp-C``` folder, which each contains the subfolders (i.e., the four new classes), ```cardboard```, ```glass```, ```metal``` and ```plastic``` with the images from the old subfolders as described in the data section. This is the data that will be used for training the classifier. 

Your repository should now look like this:

![image](https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/4cb445f8-14f3-48aa-9bb2-ef843caba106)

#### Setup
First, ensure that you have installed the *venv* package for Python (if not, run ```sudo apt-get update``` and ```sudo apt-get install python3-venv```). 

To set up the virtual environment, run ```bash setup.sh``` from the terminal (again, from the main folder).

#### Run code
To run the code, you can do the following:

##### Run script(s) with predefined arguments

From the terminal (and still in the main folder) type ```bash run.sh``` which activates the virtual environment and runs *six* classifier scripts, three scripts with different augmentation levels (none, low, high) using balanced data, and three scripts with same three augmentation levels using imbalanced data. All six models are run with 10 epochs. The results from this run can be seen in the ```out``` folder and is described in the **Results** section.

**NB:** The code may take several hours to run, depending on the processing power of your machine. 

##### Define arguments yourself

From the terminal, activate the virtual environment and run the script(s) with the desired arguments:

```
# activate virtual environment
source env/bin/activate

# run script
python3 src/classify.py --epochs <epochs> --balance <balance> --augmentation_level <augmentation_level> 
```

**Arguments:**

- **epochs:** number of epochs to run the model for. Default: 10
- **balance**: whether to balance the data by using weighted classes or run the model the original data. Must be either 'balanced' for using balanced data or 'imbalanced' to use imbalanced data. Default: 'imbalanced'
- **augmentation_level**: Level of augmentation to apply to the data. Must be 'none', 'low', or 'high'. Default: 'none'

### Results

#### History plots


**Balanced data, no augmentation**

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/195cf649-5107-4ed0-9c92-093b21256a9d" width="500">


**Balanced data, low augmentation**

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/bb011cb5-7dd1-4c55-8489-46e4aceb4042" width="500">


**Balanced data, high augmentation**

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/dbff8f4c-0bd0-4a50-8579-7c8c9663e5f7" width="500">


**Imbalanced data, no augmentation**

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/62f46199-a6cf-42bf-96cd-bb79939f3310" width="500">


**Imbalanced data, low augmentation**

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/1301d68b-697f-46f2-9c2a-b208ce3ef488" width="500">

**Imbalanced data, high augmentation**

<img src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/0ce942fa-e148-4cbb-a369-f07f9a5c9f55" width="500">

#### Classification reports

For a better overview, the classification reports for the imbalanced and balanced results have been gathered in a table. To see each classification report seperately, go to the ```out``` folder.

##### Imbalanced data

<img width="767" alt="Skærmbillede 2023-05-22 kl  16 33 58" src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/87e66e8b-880a-44de-a142-52cb13be2a3c">

##### Balanced data

<img width="767" alt="Skærmbillede 2023-05-22 kl  16 43 05" src="https://github.com/louisebphansen/assignment4-WARP-classification/assets/75262659/68651d2c-fe90-4804-98f5-b9836c7d70cb">


Looking at the history plots, it is clear that whether the data is balanced or not makes a difference. When balancing the data, the curves in the history plots follow each other more nicely, i.e., training and validation loss decreases somewhat similarly, and training and validation accuracy increases similarly. For the imbalanced data however, the accuracy curves especially could indicate that the model does not generalize very well, as the validation accuracy is not improving along the training accuracy. It does not show signs of being any of the imbalanced models to be very good. This is also evident from the classification reports, where all three imbalanced models actually show a fair perfomance when looking only at accuracy and weighted average F1 scores. However, when looking closely at the classes seperatly, it becomes clear that the models are very bad, as they are only performing well on the 'plastic' class. It is likely that the history plots and classification reports yield strange results due to the high imbalanced of number of samples in the data, and that they only have learned to predict the ‘plastic’ class.

When inspecting the impact different levels of data augmentation has on the balanced dataset, the 'no augmentation' and 'low augmentation' both yield nice looking history plots. The 'high' augmentation balanced model, on the other hand, has lower levels of validation accuracy and higher levels of loss, which indicates that the model does not generalize very well to new data. The classification reports for the balanced models show that even though the balanced models have a lower overall accuracy, the F1 scores are distributed more equally, yielding a higher macro average for all the models. The more 'distributed accuracy' comes at the cost of the performance of the 'plastic' class, however, which has decreased somewhat compared to the imbalanced models. When looking at augmentation levels, the high augmentation gives worse performance compared to no or low augmentation. This could indicate that using too much augmentation adds non-significant noise to the data, which does not improve the model's performance. There is not a very big difference between the no augmentation model and the low augmentation model, which again could indicate that for this scenario, using data augmentation does not help with performance. Although the balanced models are arguably better than the imbalanced ones, they are still not very good. Their performance would not suffice in a real-world scenario. The best solution to this problem would be to gather more data for the classes with few samples.





