# Two Layer Neural Network
# 0. [Env Setup]
* Python 3.8.8 
* Packages Installed:
    * numpy (Ver 1.20.1)
    * matplotlib (Ver 3.3.4) 
    * pandas (Ver 1.2.3)

* If wanna run the src code on Google Colab, find them in the following links:
    * [(Google Colab) Two-Layer-Neural-Network.ipynb](https://colab.research.google.com/drive/19sQorVGHmw4472ZVsIw7NyVCLALjHzVn?usp=sharing)

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

### 2.2 [Model Design]
* Input Layer: It receives a data vector (X) whose length is 769.
* Hidden Layer:
    * The 1st hidden layer is the Weight Matrix (W1) with the shape of (769, 300). (From X to Z1)
    * The 2nd hidden layer is the Weight Matrix (W2) with the shape of (300, 50). (From A1 to Z2)
* Output Layer: Followed by Softmax operation (From a2 to Ypred), it outputs a possibility vector whose length is 50.
* Note: I apply activation function (sigmoid) on Z1 (1, 300) and Z2 (1, 300). By doing so, I get a1 (1, 50) and a2 (1, 50).
<br>

![Structure of Perceptron](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW2/imgs/Two-Layer-NN.png)

### 2.3 [Training Procedures]
* Weight Initialization: Che
* Forward Propagation: Check (nn.py) ```def forward_pass(self, X, W1, W2):...```
* Calculate Loss: Cross Entropy
* Backward Propagation:
* Update Weight:
<br>

# 3. Evaluate performance
### 3.1 [Training Phase]

### 3.2 [Testing Phase]