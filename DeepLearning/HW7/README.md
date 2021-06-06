# Semantic Segmentation
* [CodaLab Link](https://competitions.codalab.org/competitions/30993) Competition: Woodscape Fisheye Semantic Segmentation for Autonomous Driving | CVPR 2021 OmniCV Workshop Challenge
* Codalab ID: Ratherman

# Outline
* Results: 
* Src Code
* Ref Link
* Datasets/ Labels
* Step 01: Visualize Images and Masks
* Step 02: Define Custom Datasets
* Step 03: Define Dataloader
* Step 04: Define Structure
* Step 05: Training
* Step 06: Draw Loss Curve
* Step 07: Testing

## Results
* First Try:
   * Results: `score:0.34, mIoU:0.31, mAcc:0.46 (Rank: 17/17)`
   * GPU: GTX 1600
   * Spend: About 420 min
   * Epoch: 60
   * Learning Rate: 1e-4 (Become 1e-5 at 25; become 1e-6 at 50)
   * Batch Size: 12
   * Input Image: 192 x 192 (Mainly due to the limitation of GPU, so I scaled it down to 192 x 192)
   * Display (Left: Input Image; Middle: Ground Truth Mask; Right: Predicted Mask)
   * <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/display.png" width="750">
   * Loss Curve
   * <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/loss.png" width="750">
* Second Try:
   * Results: `Still Training`
   * GPU: Tesla P100
   * Spend: About 280 min
   * Epoch: 20
   * Learning Rate: 1e-4 (Become 1e-5 at 10; become 1e-6 at 15)
   * Batch Size: 11
   * Input Image: 512 x 512 (This time is much larger than the one in 1st try!)

## Src Code
There are two src codes. One is on github, and another is on google colab.
* Github src code: https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/unet.ipynb
* Google Colab src code: https://colab.research.google.com/drive/14vcRp54mPCniRnPUPuw4U3r2dFBh-3_R#scrollTo=bA1CNi6M4SAT

## Ref Link
* The repo of "[usuyama/pytorch-unet](https://github.com/usuyama/pytorch-unet)" really helps me out. It shows how to use UNet to do Image Segmentation.
* Some modifications are needed to complete this multi-class image segmentation task. And the person named "[ptrblck](https://discuss.pytorch.org/u/ptrblck)" on PyTorch Forum describes clearly on how to make the task work. The following links show the posts that help me a lot:
   1. [Loss function for segmentation models](https://discuss.pytorch.org/t/loss-function-for-segmentation-models/32129/4)
   2. [Multiclass segmentation U-net masks format](https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/5)

## Datasets/Lables
```python
map = {
   0: "void",
   1: "road",
   2: "lanemarks",
   3: "curb",
   4: "pedestrians",
   5: "rider",
   6: "vehicles",
   7: "bicycle",
   8: "motorcycle",
   9: "traffic_sign"
}
```
* Expected output for each pixel is `[0, 0, 0]`, `[1, 1, 1]`, ... or `[9, 9, 9]`.
* Shape of the each output mask should be `(width, height, channel)`.

## Step 01: Visualize Images and Masks
## Step 02: Define Custom Datasets
## Step 03: Define Dataloader
## Step 04: Define Structure
## Step 05: Training
## Step 06: Draw Loss Curve
## Step 07: Testing