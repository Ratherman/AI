# Semantic Segmentation
* [Competition: Woodscape Fisheye Semantic Segmentation for Autonomous Driving | CVPR 2021 OmniCV Workshop Challenge](https://competitions.codalab.org/competitions/30993)
* Codalab ID: Ratherman

# Outline
* [Results](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#results)
* [Src Code](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#src-code)
* [Ref Link](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#ref-link)
* [Datasets/ Labels](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#datasetslables)
* [Step 01: Visualize Images and Masks](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#step-01-visualize-images-and-masks)
* [Step 02: Define Custom Datasets](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#step-02-define-custom-datasets)
* [Step 03: Define Structure](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#step-03-define-structure---classic-unet)
* [Thoughts](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#thoughts)

# Results
* First Try:
   * Results: `score:0.34, mIoU:0.31, mAcc:0.46 (Rank: 17/17)`
   * GPU: GTX 1600
   * Spend: About 420 min (7hr)
   * Epoch: 60
   * Learning Rate: 1e-4 (Become 1e-5 at 25; become 1e-6 at 50)
   * Batch Size: 12
   * Input Image: 192 x 192 (Mainly due to the limitation of GPU, so I scaled it down to 192 x 192)
   * Display (Left: Input Image; Middle: Ground Truth Mask; Right: Predicted Mask)
   * <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/display.png" width="750">
   * Loss Curve
   * <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/loss.png" width="750">
* Second Try:
   * Results: `score:0.38, mIoU:0.35, mAcc:0.50 (Rank: 17/17)`
   * GPU: Tesla P100
   * Spend: About 900 min (15hr)
   * Epoch: 60
   * Learning Rate: 1e-4 (Become 1e-5 at 25; become 1e-6 at 50)
   * Batch Size: 11
   * Input Image: 512 x 512 (This time is much larger than the one in 1st try!)
   * Display (Left: Input Image; Middle: Ground Truth Mask; Right: Predicted Mask)
   * <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/display3.png" width="750">
   * Loss Curve
   * <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/loss_2.png" width="750">

# Src Code
There are two src codes. One is on github, and another is on google colab.
* Github src code: https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/unet.ipynb
* Google Colab src code: https://colab.research.google.com/drive/14vcRp54mPCniRnPUPuw4U3r2dFBh-3_R#scrollTo=bA1CNi6M4SAT
* The whole src code does the following:
   1. Resize original images and masks. Make them smaller. (ex: 512 x 512 or 192 x 192)
   2. Train
   3. Test
   4. Resize predicted mask back to the size of 966 x 1280.
   * <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/task.png" width="750">

# Ref Link
* The repo of "[usuyama/pytorch-unet](https://github.com/usuyama/pytorch-unet)" really helps me out. It shows how to use UNet to do Image Segmentation.
* Some modifications are needed to complete this multi-class image segmentation task. And the person named "[ptrblck](https://discuss.pytorch.org/u/ptrblck)" on PyTorch Forum describes clearly on how to make the task work. The following links show the posts that helps me a lot:
   1. [Loss function for segmentation models](https://discuss.pytorch.org/t/loss-function-for-segmentation-models/32129/4)
   2. [Multiclass segmentation U-net masks format](https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/5)
* [Paper Link](https://arxiv.org/abs/1505.04597) U-Net: Convolutional Networks for Biomedical Image Segmentation

# Datasets/Lables
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
* Shape of the each output mask should be `(width = 966, height = 1280, channel = 3)`.

# Step 01: Visualize Images and Masks
* Left: Input Image; Right: Ground Truth Mask
* <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/display2.png" width="750">

# Step 02: Define Custom Datasets
* Note: Use `ToTensor` to noramlize image value.
* Note: I am using `nn.CrossEntropyLoss()` as the loss function, its expected input for mask and image are listed below:
   * Mask: `(batch size, width, height)`
   * Image: `(batch size, channel, width, height)`
   * See `def __getitem__(self, index)`.
   * Using `cv.INTER_NEAREST` for `msk_resize` is to avoid the value of mask becomes floating numbers.
   ```python
   import cv2 as cv
   from torch.utils.data import Dataset, DataLoader
   from torchvision import transforms, datasets, models

   class RoadDataset(Dataset):
      
      def __init__(self, width, height, path_to_imgs, path_to_mask, transform = None):
        
        self.height = height
        self.width = width
        self.path_to_img = path_to_imgs
        self.path_to_mask= path_to_mask
        
        self.train_imgs = os.listdir(path_to_imgs)
        self.train_mask = os.listdir(path_to_mask)
        
        self.length = len(self.train_imgs)
        self.transform = transform
        
    def __getitem__(self, index):
        
        img = cv.imread(self.path_to_img + self.train_imgs[index])
        msk = cv.imread(self.path_to_mask + self.train_mask[index])
        
        img_resize = cv.resize(img, (self.width, self.height), interpolation = cv.INTER_CUBIC)
        msk_resize = cv.resize(msk, (self.width, self.height), interpolation = cv.INTER_NEAREST)
        
        msk_transpose = msk_resize.transpose((2, 0, 1))
        msk_one_channel = msk_transpose[0]
        
        if self.transform:
            img_tensor = self.transform(img_resize)
        
        return (img_tensor, msk_one_channel)
        
    def __len__(self):
        return self.length
   ```

# Step 03: Define Structure -> UNet (with minor modification)
* <img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW7/imgs/UNet_Architecture.png" width="750">
* Basic Block:
   ```python
   import torch
   import torch.nn as nn

   def double_conv(in_channels, out_channels):
      return nn.Sequential(
         nn.Conv2d(in_channels, out_channels, 3, padding=1),
         nn.ReLU(inplace=True),
         nn.Conv2d(out_channels, out_channels, 3, padding=1),
         nn.ReLU(inplace=True)
      ) 
   ```
* All Blocks: 
   * A Encoder-Decoder Structure
   * Fully Convolutional Network
   * 
   ```python
   class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out
   ```

# As For Training, Evaluation, Testing, Please directly see src code.
* [src code](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW7#src-code)

# Thoughts:
* It's still possible to keep the image resolution even with the limited GPU resources, which is to tile the large image.
* By simply providing input images with more resolution, the overall performance could be higher.
* Future plan:
   * For model-centric approach: It might be helpful by replace the backbone.
   * For data-centric approach: It might be helpful by tiling the input images.