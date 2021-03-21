# Multi-Class-Perceptron

## Introduction:
* [Data Source](https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr) (includeing training, validation and testing dataset.)
* We have 50 types of animals, and each type contains 1300 images. The goal for our trained model, i.e. Perceptron, is to recognize the animals in the future as long as we provide it an image.

## Train Model
1. Image Pre-processing
    * Resize Image: Every Image has different scale, for sake of fairness, I resize each image into (256,256).
        * (see function **create_COH_dataset** in **Multi-Class-Perceptron.py**)
    * Feature Extraction: Use [Color Histogram](https://en.wikipedia.org/wiki/Color_histogram) to extract the features.
        * (see function **color_of_histogram** in **Multi-Class-Perceptron.py**)
    * (Optional) Save transformed datasets: Because RF and XGBoost needs the same transformed datasets, I decided to save them in csv format.
        * (see function **save_dataset** in **Multi-Class-Percetron.py**)
2. Model Design
    * Input Layer:
        * 
    * Weight Matrix:
        * 
    * Output Layer:
        *  
3. Loss Function Measurance
    * Cross Entropy: Use cross entropy to measure the current performance of the model.
4. Update the current Model (Gradient Descent and Back Propagation)
![Structure of Perceptron](https://github.com/Ratherman/AI/blob/main/DeepLearning/structure_of_perceptron.png)

## Evaluate performance
### [Training Phase]
* Once the training finishes one epoch, it concludes the current performance on both **training** and **validation** datasets.
* By PERFORMANCE, I mean the "top-1 accuracy" and "top-5 accuracy" of the datasets.

### [Testing Phase]
* Once the training phase ends, it concludes the final performance on both **validation** and **testing** datasets.
