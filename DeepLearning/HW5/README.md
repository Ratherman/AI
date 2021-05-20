
# Introduction
* Competition main entrance: https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4?focus=intro
* In HW5, we are going to cope with AOI - task. It's basically a classification problem where the input would be a PNG file and the output would be the desired label.

# Note
* In img/, the training/ validation accuracies are stored for every experiment below.

# Final Result: After Experiment with val curve with various model (LeNet5, AlexNet, ResNet18), I decided to use LeNet5 without data augmentation. (See A1 ~ A3 for detail)
* LeNet5-like Model (Use 940 MX): Train Acc. 99%, Test Acc. 95.31442 %
    * No Data Augmentation    
    * Optimizer: RMSprop(learning rate = 1e-4)
    * Val Split: 0.0 (All images used for training)
    * Batch Size: 64
    * Epoch: 50
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 24,595,092
    * Forward/ backward pass size: 32.81 MB
    * Params size: 93.82 MB
    * Estimated Total Size: 129.94 MB

* Thoughts:
    * 在仔細觀察 Training Dataset 發現/突然考慮到，原有的 Data Augmentation 說不定會切到 Edge Defect 的 Image 導致模型 confused 所以這裡的 Data Augmentation 改成以下這樣。
    * 512x512 → H_Flip(p=0.5) → V_Flip(p=0.5) → Normalize(mean=0.5, std.=0.5)
    * Note: 下面有使⽤ Data Augmentation 的實驗維持原本的設定，沒有另外再做測試了。

* LeNet5-like Model (Use GTX 1660): Train Acc. 98%, Test Acc. 96.17755 %
    * Use Data Augmentation: 512x512 → H_Flip(p=0.5) → V_Flip(p=0.5) → Normalize(mean=0.5, std.=0.5)
    * Optimizer: RMSprop(learning rate = 1e-4)
    * Val Split: 0.0 (All images used for training)
    * Batch Size: 64
    * Epoch: 50
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 24,595,092
    * Forward/ backward pass size: 32.81 MB
    * Params size: 93.82 MB
    * Estimated Total Size: 129.94 MB

# Grades with val curve (8 experiments)
* Experiment A1: LeNet5-like Model V1 (Use GTX 1660): Train Acc. 98%, Val Acc. 93%, Test Acc. 92.50308% || Spend 1561 sec.
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

* Experiment A2: LeNet5-like Model V2 (Use GTX 1660): Train Acc. 92%, Val Acc. 91%, Test Acc. 90.55487% || Spend 2356 sec.
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
    * Thoughts:
        1. Data Augmentation helps decrease the gap between train acc. and val acc.
        2. The overall performance didn't get better.

* Experiment A3: LeNet5-like Model V3 (Use GTX 1660): Train Acc. 93%, Val Acc. 92%, Test Acc. 80.19728% || Spend 3794 sec.
    * Use Data Augmentation: 512x512 -> Resize 712x712 -> CenterCrop 612x612 -> RandomCrop 512x512 -> H_Flip(p=0.5) -> V_Flip(p=0.5) -> Normalize(mean=0.5, std.=0.5) <---- add data augmentation
    * Optimizer: RMSprop(learning rate = 1e-4)
    * Val Split: 0.2
    * Batch Size: 8
    * Epoch: 50
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 24,595,092
    * Forward/ backward pass size: 32.81 MB
    * Params size: 93.82 MB
    * Estimated Total Size: 129.94 MB
    * Thoughts:
        1. Test Acc. somehow decrease to 8x % which might suggest that current methods of data augmentation may hurt the performance.

* Experiment B1: AlexNet-like Model V1 (Use 940 MX): Train Acc. 85%, Val Acc. 80%, Test Acc. 82.26880% || Spend 6475 sec.
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

* Experiment B2: AlexNet-like Model V2 (Use 940 MX): Train Acc. 87%, Val Acc. 84%, Test Acc. N/A% || Spend 6475 sec.
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

* Experiment B3: AlexNet-like Model V3 (Use GTX 1660): Train Acc. 80%, Val Acc. 82%, Test Acc. N/A% || Spend 2440 sec.
    * Use Data Augmentation: 512x512 -> Resize 712x712 -> CenterCrop 612x612 -> RandomCrop 512x512 -> H_Flip(p=0.5) -> V_Flip(p=0.5) -> Normalize(mean=0.5, std.=0.5) <---- add data augmentation
    * Optimizer: RMSprop(learning rate = 1e-5) <---- change back to 1e-5, same as v1
    * Val Split: 0.2
    * Batch Size: 32
    * Epoch: 30
    * Input Tensor: 3 x 512 x 512 (Input Size: 3.00 MB)
    * Trainable Params: 44,424,006
    * Forward/ backward pass size: 44.55 MB
    * Params size: 169.46 MB
    * Estimated Total Size: 217.02 MB
    * Thoughts:
        1. Data Augmentation indeed helps to decrease the gap between train acc. and test acc.
        2. But the overall performance still didn't get better.
        3. Didn't test it for test acc. because train acc didn't surpass 90%.

* Experiment C1: ResNet-like Model V1 (Use CPU): Train Acc. 93%, Val Acc. 91%, Test Acc. N/A || Spend 10403.83 sec.
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

* Experiment C2: ResNet-like Model V2 (Use 940 MX): Train Acc. 90%, Val Acc. 89%, Test Acc. N/A || Spend 1615.18 sec.
    * Use Data Augmentation: 256x256 -> Resize 456x456 -> CenterCrop 356x356 -> RandomCrop 256x256 -> H_Flip(p=0.5) -> V_Flip(p=0.5) -> Normalize(mean=0.5, std.=0.5)
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
        1. Test Acc. is very low --> 30% ~ 50%. Could be issue of overfitting.
        2. Didn't test it for test acc because the acc is predictably low.
        3. Even with the help of data augmentation, the prediction in testing phase is still strange.
        4. It might be due to the imbalance of classes in training dataset, so the model tends to assign class 1 as label.