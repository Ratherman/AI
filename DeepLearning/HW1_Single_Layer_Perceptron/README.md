# Multi-Class-Perceptron
# 0. [Env Setup]
* Python 3.8.8 
* Packages Installed:
    * numpy (Ver 1.20.1)
    * matplotlib (Ver 3.3.4) 
    * pandas (Ver 1.2.3)
    * opencv-python (Ver 4.5.1) 
    * scikit-learn  (Ver 0.24.1)
    * xgboost (Ver 1.3.3)
* If wanna run the src code on Google Colab, find them in the following links:
    * [(Google Colab) Multi-Class-Perceptron.ipynb](https://colab.research.google.com/drive/1J_7woxV6hS5JVgpAGht-3Frbg4heDeE3?authuser=2#scrollTo=fFvesDo5Ra2I)
    * [(Google Colab) Random Forest and XGBoost.ipynb](https://colab.research.google.com/drive/1YaMmLyFjPLGtt2uAehAegyTajFBzreHT?authuser=2#scrollTo=qDcFxtDZIBfX)

# 1. Intro:
### 1.1 [Download Dataset]
* [Data Source](https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr) (including training, validation and testing dataset.)
### 1.2 [Dataset Intro]
* We have 50 types of animals, and each type contains 1300 images. The goal for our trained model, i.e. Perceptron, is to recognize the animals in the future as long as we provide it an image.

# 2. Train Model
### 2.1 [Image Pre-processing]
* Resize Image: Every Image has different scale, for sake of fairness, I resize each image into (256,256,3).
    * (Function **create_COH_dataset**)
* Feature Extraction: Use [Color Histogram](https://en.wikipedia.org/wiki/Color_histogram) to extract the features.
    * (Function **color_of_histogram**)
* (Optional) Save transformed datasets: Because RF and XGBoost needs the same transformed datasets, I decided to save them in csv format.
    * (Function **save_dataset**)

### 2.2 [Model Design]
* Input Layer: It receives a data vector (X) whose length is 769.
* Weight Matrix: The matrix (W) contains all trainable parameters whose shape is (769,50).
* Output Layer: Followed by Softmax operation (From Z to S), it outputs a possibility vector whose length is 50.
<br>

![Structure of Perceptron](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW1_Single_Layer_Perceptron/imgs/structure_of_perceptron.png)

### 2.3 [Loss Function Measurement]
* Forward Propagation: Pass the data vector forward to get the current result.
    * (Function **Softmax**)
* Cross Entropy: Use cross entropy to measure the current performance of the model.
    * (Function **CrossEntropy**)
<br>

![Epoch and Loss](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW1_Single_Layer_Perceptron/imgs/Perceptron_Train_Loss.png)

### 2.4 [Update the current Model]
* Back Propagation and Gradient Descent
    * (Codes between 234 and 300)

# 3. Evaluate performance
### 3.1 [Training Phase]
* Once the training finishes one epoch, it concludes the current performance on both **training** and **validation** datasets.
* By PERFORMANCE, I mean the "top-1 accuracy" and "top-5 accuracy" of the datasets.
<br>

![Top 1 and Top 5 of Train and Val](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW1_Single_Layer_Perceptron/imgs/Perceptron_Train_Val_Acc_Record_Each_Epoch.png)
### 3.2 [Testing Phase]
* Once the training phase ends, it concludes the final performance on both **validation** and **testing** datasets.
* The following image shows the performance of **Perceptron**.
<br>

![Top 1 and Top 5 of Val and Test](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW1_Single_Layer_Perceptron/imgs/Perceptron_Test_Val_Acc.png)
* The following image shows the performance of **Random Forest**.
<br>

![Random Forest of Val and Test](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW1_Single_Layer_Perceptron/imgs/RF_Test_Val_Acc.png)
* The following image shows the performance of **XGBoost**.
<br>

![XGBoost of Val and Test](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW1_Single_Layer_Perceptron/imgs/XGBoost_Test_Val_Acc.jpg)