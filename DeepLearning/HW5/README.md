# Introduction
In HW5, we are going to cope with AOI - task. It's basically a classification problem where the input would be a PNG file and the output would be the desired label.

# Grades
1. LeNet-like Model v1: Train Acc 84%, Test Acc. 80%
    * best train acc: 91%
    * Learning Rate: 1e-3
    * epoch: 15
    * tensor: 512 x 512 x 3

2. LeNet-like Model v2: Train Acc 99%, Test Acc. 95%
    * best train acc: 99%
    * Learning Rate: 1e-4
    * epoch: 30
    * Input tensor: 512 x 512 x 3
    * Total/ Trainable params: 24,595,092
    * Input size (MB): 3.00
    * Forward/backward pass size (MB): 32.81
    * Params size (MB): 93.82
    * Estimated Total Size (MB): 129.64
    * Spend Time: 4405.28 (sec)

3. LeNet-like Model v3: Train Acc. 98%, Test Acc. 95%
    * best train acc: 98%
    * Learning Rate: 1e-4
    * epoch: 30
    * Input tensor: 512 x 512 x 3
    * Total/ Trainable params: 24,595,092
    * Input size (MB): 3.00
    * Forward/backward pass size (MB): 32.81
    * Params size (MB): 93.82
    * Estimated Total Size (MB): 129.64
    * Use:
        * Normalization (only difference between v2 and v3): input = input / 255.
    * Spend Time: 5242.81 (sec)

4. ResNet-like Model v1: Train Acc. XX%, Test Acc. XX%
    * best train acc: XX%
    * Learning Rate: 1e-4
    * epoch: 30
    * Input tensor: 512 x 512 x 3
    * Total/ Trainable params: XX,XXX,XXX
    * Input size (MB): X.XX
    * Forward/backward pass size (MB): XX.XX
    * Params size (MB): XX.XX
    * Estimated Total Size (MB): XXX.XX
    * Use:
        * Normalization
    * Spend Time: XXXX.XX (sec)


# Labels
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

