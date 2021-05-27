import csv

result_json= [
"""
# paste the result.json here
# demo:

{
 "frame_id":1, 
 "filename":"data/test/2010111918.jpg", 
 "objects": [ 

 ] 
}, 
...
"""
]

def main(result_json):
    with open('results_v1.csv', 'w', newline='') as csvfile:
        
        fieldnames = ['image_filename', 'label_id', 'x', 'y', 'w', 'h', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for ele in result_json:
        
            filename = get_filename(ele["filename"])
        
            if ele["objects"] != []:
                for bbox in ele["objects"]:
                    label_id, x, y, w, h, confidence = get_bbox_info(bbox)
                
                    print(filename, label_id + 1, x, y, w, h, confidence)
                    writer.writerow({'image_filename': filename, 'label_id': label_id + 1, 'x': x, 'y': y, 'w': w, 'h': h, 'confidence': confidence})

def get_filename(file_path):
    filenames = file_path.split("/")
    filename = filenames[-1]
    return filename

def get_bbox_info(obj):
    label_id = obj["class_id"]
    cx = obj["relative_coordinates"]["center_x"]
    cy = obj["relative_coordinates"]["center_y"]
    w = obj["relative_coordinates"]["width"]
    h = obj["relative_coordinates"]["height"]
    confidence = obj["confidence"]
    
    cx = cx * 400
    cy = cy * 300
    w = w * 400
    h = h * 300
    
    x = cx - w/2
    y = cy - h/2
    
    x = round(x)
    y = round(y)
    w = round(w)
    h = round(h)
    
    return label_id, x, y, w, h, confidence

main(result_json)