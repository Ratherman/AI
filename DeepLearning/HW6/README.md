# Object Detection
## 尋找病媒蚊孳生源-積水容器影像物件辨識
* AIdea Entry: https://aidea-web.tw/topic/cc2d8ec6-dfaf-42bd-8a4a-435bffc8d071

## Ref Link
* https://github.com/argusswift/YOLOv4-pytorch/tree/d0a3b6553c22a7e45e218fb1b82465367e833792

## Datasets/Labels:
* Train Datasets: 2671
* Annotations Format: PASCAL VOC

```
map = {
    1: aquarium,
    2: bottle,
    3: bowl,
    4: box,
    5: bucket,
    6: plastic_bag,
    7: plate,
    8: styrofoam,
    9: tire,
    10: toilet,
    11: tub,
    12: washing_machine,
    13: water_tower
}
```
## Step 01: Convert VOC Format to YOLO Format
* Python code: convert_voc_to_yolo_format
* re-arrange the value of label_id into [0, 12].
* After converting, the result looks like this:
```
# label_id, cx, cy, w, h (ex: 20080110.jpg)
3 0.14375 0.5458333333333333 0.19375 0.26666666666666666
3 0.7109375 0.75625 0.396875 0.3458333333333333
```

## Step 02: Generate train.txt
* Python code: theAIGuysCode/YOLOv4-Cloud-Tutorial/yolov4/generate_train.py
* In this step, I used the code from [Github repo: theAIGuysCode](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)
* train.txt describes all images we used in training phase.
```
# the content in train.txt looks like:
data/obj/20130304.jpg
data/obj/201808104.jpg
data/obj/201604204.jpg
...
```

## Step 03: Preparation & Configuration
* Config.: mainly followed the instructions from theAIGuys. [[YouTube Link](https://www.youtube.com/watch?v=mmj3nxGT2YQ)]
* There are at least 3 files need to be configured.(1) `obj.names` (2) `obj.data` (3) `yolov4-obj.cfg`
#### (1) obj.names: 
* Note: basically one class per line
```
aquarium
bottle
bowl
box
bucket
plastic_bag
plate
styrofoam
tire
toilet
tub
washing_machine
water_tower
```
#### (2) obj.data
* Note 1: `train.txt` is exactly the result I got after finishing step 02.
* Note 2: `backup` means where to save the weight every 100 epoch. I decided to save the weight in the root of Google VM because when I chose to save weights into my google drive, it always crashed, which might be due to heavy io between vm and drive. Therefore, I decided to save the weight directly in VM root. And save the weights manually by clicking download option every several hour. 
```
classes = 13
train = data/train.txt
valid = data/train.txt
names = data/obj.names
backup = ../
```
#### (3) yolov4-obj.cfg
* Note 1: There are so about 1000 options related to hyper-parameters in `yolov4-obj.cfg`.
* Note 2: I will mentioned a few in the following.
```
#Traing
batch=128 # Setting this to 256 would run out of memory, so set this to 128.
...
width=416  # Both width and height have to be multiple of 32. I set the width to the one surpass the image width (400), which is 416.
height=320 # I set the height to the one surpass the image height (300), which is 312.
...
momentum=0.949 # Default
decay=0.0005 # Default
...
learning_rate=0.001 # Default
... 
max_batches = 10000 / 26000 # This means how many batches will be trained in training phase.
policy=steps # Learning rate will be multiply by a coeficient (0.1 per the follow) at certain steps.
steps=8000, 9000 / 20800, 23400 # Trigger step at these two steps. (0.8 x max_batches, 0.9 x max_batches)
scales=.1,.1
...
classes=13 # We have 13 different classes. Note that the algorithm accepts the range of label id from 0 to 12. (see step 01)
```