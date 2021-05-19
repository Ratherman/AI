# Introduction
In HW5, we are going to cope with AOI - task. It's basically a classification problem where the input would be a PNG file and the output would be the desired label.

# Grades
* A1: LeNet5-like Model V1 (Use GTX 1660): Train Acc. 98%, Val Acc. 93%, Test Acc. 92.50308% || Spend 1561 sec.
    * Not Use Data Augmentation
    * Optimizer: RMSprop(learning rate = 1e-4)
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 30
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 24,595,092
    * Forward/ backward pass size: 32.81 MB
    * Params size: 93.82 MB
    * Estimated Total Size: 129.94 MB

* A2: LeNet5-like Model V2 (Use GTX 1660): Train Acc. 92%, Val Acc. 91%, Test Acc. 90.55487% || Spend 2356 sec.
    * Use Data Augmentation: 512x512 -> Resize 712x712 -> CenterCrop 612x612 -> RandomCrop 512x512 -> H_Flip(p=0.5) -> V_Flip(p=0.5) -> Normalize(mean=0.5, std.=0.5) <---- add data augmentation
    * Optimizer: RMSprop(learning rate = 1e-4)
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 30
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 24,595,092
    * Forward/ backward pass size: 32.81 MB
    * Params size: 93.82 MB
    * Estimated Total Size: 129.94 MB

* B1: AlexNet-like Model V1 (Use 940 MX): Train Acc. 85%, Val Acc. 80%, Test Acc. 82.26880% || Spend 6475 sec.
    * Not Use Data Augmentation
    * Optimizer: RMSprop(learning rate = 1e-5)
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 30
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 44,424,006
    * Forward/ backward pass size: 44.55 MB
    * Params size: 169.46 MB
    * Estimated Total Size: 217.02 MB

* B2: AlexNet-like Model V2 (Use 940 MX): Train Acc. 87%, Val Acc. 84%, Test Acc. N/A% || Spend 6475 sec.
    * Not Use Data Augmentation
    * Optimizer: RMSprop(learning rate = 1e-4) <---- Only Difference Between B1 and B2
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 30
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 44,424,006
    * Forward/ backward pass size: 44.55 MB
    * Params size: 169.46 MB
    * Estimated Total Size: 217.02 MB
    * Thoughts: 
        1. After increasing its learning rate, the acc./ loss curves fluctuates strongly. Not good.
        2. Didn't test it for test acc. because both acc. fluctuates strongly.

* B3: AlexNet-like Model V3 (Use GTX 1660): Train Acc. xx%, Val Acc. xx%, Test Acc. xx.xxxxx% || Spend xxxx sec.
    * Use Data Augmentation: 512x512 -> Resize 712x712 -> CenterCrop 612x612 -> RandomCrop 512x512 -> H_Flip(p=0.5) -> V_Flip(p=0.5) -> Normalize(mean=0.5, std.=0.5) <---- add data augmentation
    * Optimizer: RMSprop(learning rate = 1e-5) <---- change back to 1e-5, same as v1
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 30
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 44,424,006
    * Forward/ backward pass size: 44.55 MB
    * Params size: 169.46 MB
    * Estimated Total Size: 217.02 MB

* C1: ResNet-like Model V1 (Use CPU): Train Acc. 93%, Val Acc. 91%, Test Acc. N/A || Spend 10403.83 sec.
    * Not Use Data Augmentation
    * Optimizer: RMSprop(learning rate = 1e-4)
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 10
    * Input Tensor: 3 x 256 x 256 (Input Size: Not Estimated Using CPU)
    * Trainable Params: Not Estimated Using CPU
    * Forward/ backward pass size: Not Estimated Using CPU
    * Params size: Not Estimated Using CPU
    * Estimated Total Size: Not Estimated Using CPU
    * Thoughts: 
        1. Test Acc. is very low --> 30% ~ 50%. Could be issue of overfitting.
        2. Didn't test it for test acc because the acc is predictably low.

* C2: ResNet-like Model V2 (Use 940 MX): Train Acc. xx%, Val Acc. xx%, Test Acc. N/A || Spend xxxxx.xx sec.
    * Use Data Augmentation: 512x512 -> Resize 712x712 -> CenterCrop 612x612 -> RandomCrop 512x512 -> H_Flip(p=0.5) -> V_Flip(p=0.5) -> Normalize(mean=0.5, std.=0.5)
    * Optimizer: RMSprop(learning rate = 1e-4)
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 10
    * Input Tensor: 3 x 256 x 256 (Input Size: 0.75 MB)
    * Trainable Params: 11,179,590
    * Forward/ backward pass size: 86.00 MB
    * Params size: 42.65 MB
    * Estimated Total Size: 129.40 MB
    * Thoughts: 
        1. N/A
        2. N/A

There are 6 different labels listed below:
* 0 - Normal
* 1 - Void
* 2 - Horizontal Defect
* 3 - Vertical Defect
* 4 - Edge Defect
* 5 - Particle

# Strategies
1.  (DONE 2021/05/08) Understand the background knowledge.
2.  (DONE 2021/05/08) Get a feel of data, draw it.
3.  (DONE 2021/05/16) "PyTorch" DataLoader
5.  (DONE 2021/05/16) "PyTorch" Version of Model (MVP --> Just a simple backbone ^_^)
8.  "PyTorch" DataAugmentation
10. Recap Teacher's Approach to build a good model.
11. "Pytorch" Build a model as best as possible.
4.  "Tensorflow" DataLoader
6.  "Tensorflow" Version of Model (MVP --> Just a simple backbone ^_^)
7.  Start to use TensorBoard.
9.  "Tensorflow" DataAugmentation
12. "Tensorflow" Build a model as best as possible.

