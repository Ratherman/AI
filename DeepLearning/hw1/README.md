# Multi-Class-Perceptron

## Introduction:
* [Data Source](https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr) (includeing training, validation and testing dataset.)
* We have 50 types of animals, and each type contains 1300 images. The goal for our trained model, i.e. Perceptron, is to recognize the animals in the future as long as we provide it an image.

## Train Model
### [Image Pre-processing]
* Resize Image: Every Image has different scale, for sake of fairness, I resize each image into (256,256,3).
    * (Function **create_COH_dataset**)
* Feature Extraction: Use [Color Histogram](https://en.wikipedia.org/wiki/Color_histogram) to extract the features.
    * (Function **color_of_histogram**)
* (Optional) Save transformed datasets: Because RF and XGBoost needs the same transformed datasets, I decided to save them in csv format.
    * (Function **save_dataset**)

### [Model Design]
* Input Layer: It receives a data vector whose length is 769.
* Weight Matrix: The matrix contains all trainable parameters whose shape is (769,50).
* Output Layer: Followed by Softmax operation, it outputs a possibility vector whose length is 50.
![Structure of Perceptron](https://github.com/Ratherman/AI/blob/main/DeepLearning/structure_of_perceptron.png)

### [Loss Function Measurance]
* Forward Propagation: Pass the data vector forward to get the current result.
    * (Function **Softmax**)
* Cross Entropy: Use cross entropy to measure the current performance of the model.
    * (Function **CrossEntropy**)
### [Update the current Model]
* Back Propagation and Gradient Descent
    * (Codes between 234 and 300)

## Evaluate performance
### [Training Phase]
* Once the training finishes one epoch, it concludes the current performance on both **training** and **validation** datasets.
* By PERFORMANCE, I mean the "top-1 accuracy" and "top-5 accuracy" of the datasets.
![Top 1 and Top 5 of Train and Val](https://github.com/Ratherman/AI/blob/main/DeepLearning/Perceptron_Train_Val_Acc_Record_Each_Epoch.png)
### [Testing Phase]
* Once the training phase ends, it concludes the final performance on both **validation** and **testing** datasets.
![Top 1 and Top 5 of Val and Test](https://github.com/Ratherman/AI/blob/main/DeepLearning/Perceptron_Test_Val_Acc.png)
