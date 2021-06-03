# Semantic Segmentation
* [CodaLab Link](https://competitions.codalab.org/competitions/30993) Competition: Woodscape Fisheye Semantic Segmentation for Autonomous Driving | CVPR 2021 OmniCV Workshop Challenge

# Datasets and Lables
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

* Expected output for each pixel is [0, 0, 0], [1, 1, 1], ... or [9, 9, 9].