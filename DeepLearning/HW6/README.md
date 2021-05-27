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
* Python code: conver_voc_to_yolo_format
* re-arrange the value of label_id into [0, 12].
* After converting, the result looks like this: (ex: 20080110.jpg)
```
# label_id, cx, cy, w, h
3 0.14375 0.5458333333333333 0.19375 0.26666666666666666
3 0.7109375 0.75625 0.396875 0.3458333333333333
```
## Step 02: Generate train.txt