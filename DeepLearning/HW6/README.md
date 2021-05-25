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
    0: aquarium,
    1: bottle,
    2: bowl,
    3: box,
    4: bucket,
    5: plastic_bag,
    6: plate,
    7: styrofoam,
    8: tire,
    9: toilet,
    10: tub,
    11: washing_machine,
    12: water_tower
}
```
## Step 01: Convert VOC Format to YOLO Format
## Step 02: Generate train.txt