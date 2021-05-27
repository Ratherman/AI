# Object Detection
## 尋找病媒蚊孳生源-積水容器影像物件辨識
* AIdea Competition Entry: https://aidea-web.tw/topic/cc2d8ec6-dfaf-42bd-8a4a-435bffc8d071

## Outline:
* [Results](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#results)
* [Src Code](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#src-code)
* [Ref Link](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#ref-link)
* [Datasets/Labels](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#datasetslabels)
* [Step 01: Convert VOC Format to YOLO Format](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#step-01-convert-voc-format-to-yolo-format)
* [Step 02: Generate train.txt](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#step-02-generate-traintxt)
* [Step 03: Preparation & Configuration](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#step-03-preparation--configuration)
* [Step 04: Training](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#step-04-training)
* [Step 05: Testing](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#step-05-testing)
* [Step 06: Parse json-format result to asked-format result.](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6#step-06-parse-json-format-result-to-asked-format-result)
## Results:
* Loss Curve and Accuracy Curve During training phase
![](https://github.com/Ratherman/AI/blob/main/DeepLearning/HW6/img/first_try_10000_max_batch.png | width=250)
<img src="https://github.com/Ratherman/AI/blob/main/DeepLearning/HW6/img/first_try_10000_max_batch.png" alt="alt text" width="250">


## Src Code:
* My Github Repo: [Link](https://github.com/Ratherman/AI/tree/main/DeepLearning/HW6)
* My Google Colab Notebook: [Link](https://colab.research.google.com/drive/1AD7hIrI6Co-vlTKPbhJWN0OiUOpaML19#scrollTo=imc0NP19hLuq)

## Ref Link
* [[YouTube Link](https://www.youtube.com/watch?v=mmj3nxGT2YQ)]: YOLOv4 in the CLOUD: Build and Train Custom Object Detector (FREE GPU)
* [[Github Link](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)]: theAIGuysCode/YOLOv4-Cloud-Tutorial
* [[Google Colab Notebook Link](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing)]: YOLOv4_Training_Tutorial.ipynb

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
* Python code: `convert_voc_to_yolo_format.py`
* re-arrange the value of label_id into [0, 12].
* After converting, the `bbox_info_yyyymmdd.txt` looks like this:
```
# label_id, cx, cy, w, h (ex: 20080110.jpg)
3 0.14375 0.5458333333333333 0.19375 0.26666666666666666
3 0.7109375 0.75625 0.396875 0.3458333333333333
```

## Step 02: Generate train.txt
* Python code: [theAIGuysCode/YOLOv4-Cloud-Tutorial/yolov4/generate_train.py](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial/blob/master/yolov4/generate_train.py)
* In this step, I used the code from [Github repo: theAIGuysCode](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial)
* `train.txt` describes all images we used in training phase.
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

## Step 04: Training
* Python code (1): Mainly follow the Instructions from [[Original Google Colab Notebook](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing)]
* Python code (2): I modify it a bit for this homework. (Also for brevity) => [[My Version of Google Colab Notebook]](https://colab.research.google.com/drive/1AD7hIrI6Co-vlTKPbhJWN0OiUOpaML19?usp=sharing)
* But the most important two lines of code are:
* (1) Get the pre-trained weights for the convolutional layers.
    ```
    !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    ```
* (2) Train darknet.
    * Note 1: `data/obj.data` and `cfg/yolov4-obj.cfg` are exactly the files we setup in step 03.
    * Note 2: `-map` flag means we want to test the performance based on val. dataset (I set val. equals to train in this case) in terms of mAP.
    ```
    !./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map
    ```
* After training, the final weight, i.e., `yolov4-obj_final.weights` will be generated.

## Step 05: Testing
* Python code (1): Mainly follow the Instructions from [[Original Google Colab Notebook](https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing)]
* Python code (2): I modify it a bit for this homework. (Also for brevity) => [[My Version of Google Colab Notebook]](https://colab.research.google.com/drive/1AD7hIrI6Co-vlTKPbhJWN0OiUOpaML19?usp=sharing)
* Python code (3) [theAIGuysCode/YOLOv4-Cloud-Tutorial/yolov4/generate_test.py](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial/blob/master/yolov4/generate_test.py)
* Use python code (3) `generate_test.py` to generate `test.txt`.
* Run the following command to test the trained model and save the output (JSON format: `result_test.json`).
```
!./darknet detector test data/obj.data cfg/yolov4-obj.cfg ../yolov4-obj_final.weights -ext_output -dont_show -out result_test.json < data/test.txt
```
* The `result.json` looks like this:
```
[
{
 "frame_id":1, 
 "filename":"data/test/2010111918.jpg", 
 "objects": [ 

 ] 
}, 
{
 "frame_id":2, 
 "filename":"data/test/20090625.jpg", 
 "objects": [ 
  {"class_id":4, "name":"bucket", "relative_coordinates":{"center_x":0.487827, "center_y":0.631973, "width":0.096893, "height":0.141298}, "confidence":0.998110}, 
  {"class_id":4, "name":"bucket", "relative_coordinates":{"center_x":0.453598, "center_y":0.565197, "width":0.123445, "height":0.212162}, "confidence":0.996475}, 
  {"class_id":4, "name":"bucket", "relative_coordinates":{"center_x":0.516845, "center_y":0.678519, "width":0.051426, "height":0.092665}, "confidence":0.951427}, 
  {"class_id":4, "name":"bucket", "relative_coordinates":{"center_x":0.592881, "center_y":0.654635, "width":0.051142, "height":0.087461}, "confidence":0.657259}
 ]
}, 
 ...
]
```

## Step 06: Parse json-format result to asked-format result.
* Python Code: `parse_json_to_csv.py`
* The asked format are:
    1. CSV-file format required
    2. six col. info required: `image_filename`, `label_id`, `x`, `y`, `w`, `h`, `confidence`
* There is also one little tricky part is that I need to remember to re-arrange the lable_id into [1, 13] (from [0, 12]).