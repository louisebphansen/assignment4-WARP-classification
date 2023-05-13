# Final Project: Waste Classification using CNN's

This assignment is the final project on the Visual Analytics course on the elective in Cultural Data Science at Aarhus University. 

### Contributions

The code was created by me. However, code provided throughout the course has been reused and adapted for this project. Furthermore, some code from [WEBSITE] has been reused. 

### Project description

This project aims to apply the tools and methods taught in the course in a waste-recycling scenario. More specifically, a dataset from Kaggle, 'WaRP' (Waste Recycling Plant) has been used. I thought it would be interesting to apply image classification and computer vision tools to a real life problem to see if it could be a possible method for optimizing waste recycling. **skriv mere her, det er dårligt** .. 

### Methods

#### Data
The WaRP data was found on Kaggle, and consists of three different sub-datasets, one for image segmentation, one for image classification and one for image detection. The image classification data was used for this assignment. The data consists of 8823 images for training and **test_images** for testing. The original data contains 28 different classes, with 17 different categories of plastic bottles, three different types of glass bottles, four different types of detergent and two different types of cardboards. I thought it would be more useful in a real-life situation to use more *overall** classes for the waste. Thus, I decided to create 4 classes consisting of the 28 original subclasses. The contents of the classes are as follows:

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
