# import needed packages
import os
import sys
import glob
import numpy
from xml.etree.ElementTree import ElementTree

# Put this file around xml files
annotations = os.listdir(".")
annotations = glob.glob(str(annotations)+'*.xml')
print(len(annotations))

# Define labels
map_table = {
    "aquarium": 0,
    "bottle": 1,
    "bowl": 2,
    "box": 3,
    "bucket": 4,
    "plastic_bag": 5,
    "plate": 6,
    "styrofoam": 7,
    "tire": 8,
    "toilet": 9,
    "tub": 10,
    "washing_machine": 11,
    "water_tower": 12
}

# Excute
for i, file in enumerate(annotations):
    file_save = '../output_txt/' + file.split('.')[0]+'.txt'
    file_txt = file_save
    f_w = open(file_txt, 'w')
    
    tree = ElementTree()
    tree.parse(file)
    root = tree.getroot()
    filename = root.find('filename').text  #這裡是xml的根，獲取filename那一欄
    info = tree.find("size")
    imgw = float(info.find("width").text)
    imgh = float(info.find("height").text)
    
    for obj in root.iter('object'):
        name = obj.find('name').text   #這裡獲取多個框的名字，底下是獲取每個框的位置
        num = map_table[name]
        
        xmax = float(obj.find("bndbox").find("xmax").text)
        xmin = float(obj.find("bndbox").find("xmin").text)
        ymax = float(obj.find("bndbox").find("ymax").text)
        ymin = float(obj.find("bndbox").find("ymin").text)
        
        w = xmax-xmin
        h = ymax-ymin
        xcenter = (w/2+xmin)/imgw
        ycenter = (h/2+ymin)/imgh
        w = w/imgw
        h= h/imgh
        f_w.write(str(num) +' '+str(xcenter)+' '+str(ycenter)+' '+str(w)+' '+str(h)+'\n')