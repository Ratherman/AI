# LeNet

## Structure of LeNet (Click to see details)
![Structure of LeNet](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW3/imgs/Structure_LeNet5.png)

## Dataset:
1. MNIST Dataset [LINK](https://www.kaggle.com/c/digit-recognizer/data)
    * 10 Different Classes
    * 60000 Training Examples, 10000 Testomg Examples
    * This Dataset is the same as the "mnist.pkl" in this dir.
2. Mini-ImageNet [LINK](https://drive.google.com/file/d/1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr/view)
    * 50 Different Classes
    * 60000 (Up) Training Examples, 449 Validation Examples, 449 Testing Examples.

## Src Code:
* My_Own_Hand-Crafted_LeNet5_[Placeholder].ipynb
    * Descirption: **Didn't work well on both MNIST and Mini-ImageNet Datasets, still need to check potential bugs.**
    * My_Own_Hand-Crafted_LeNet5_MNIST.ipynb: For training on MNIST Dataset.
    * My_Own_Hand-Crafted_LeNet5_Mini-ImageNet.ipynb: For training on Mini-ImageNet Dataset.

* Github-Cloned_Hand-Crafted_LeNet5.ipynb
    * Descirption: **Works okay on MNIST but didn't work well on Mini-ImageNet Datasets, still need to fine tune the hyperparameters.**
    * I use the repo from [IQ250 Github Link](https://github.com/IQ250/LeNet-by-Numpy), it works really well on MNIST dataset. 

* Keras-Version_LeNet5.ipynb
    * Descirption: **Work okay on both MNIST and Mini-ImageNet Datasets.**
    * Use this to see the potential outcomes from adjusting hyperparameters.
* Other Resources:
    * I also study the following repos for quite a while, but I didn't put them in this repo.
    * The repo from [toxtli](https://github.com/toxtli/lenet-5-mnist-from-scratch-numpy) is a good example on how hand-crafted LeNet designed.
    * The repo from [mattwang44](https://github.com/mattwang44/LeNet-from-Scratch) has good performance on classifying MNIST dataset.