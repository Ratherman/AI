# [Input Vars]
#   1. <string> PATH_TO_DESIRED_LOCATION: It should be the directory containing (1) images/ (2) train.txt (3) test.txt (4) val.txt

# [Output Vars]
#   1. <ndarray> np_train_txt: It contains both the directory to a specific image and the related label.
#   2. <ndarray> np_test_txt: It contains both the directory to a specific image and the related label.
#   3. <ndarray> np_val_txt: It contains both the directory to a specific image and the related label.
import pandas as pd
import numpy as np
def read_metadata_files(Root_Path):
    # train.txt
    train_txt = pd.read_csv(Root_Path+"train.txt", sep=" ")
    NP_TRAIN_TXT = np.array(train_txt)
    
    # test.txt
    test_txt = pd.read_csv(Root_Path+"test.txt", sep=" ")
    NP_TEST_TXT = np.array(test_txt)
    
    # val.txt
    val_txt = pd.read_csv(Root_Path+"val.txt", sep=" ")
    NP_VAL_TXT = np.array(val_txt)
    
    print(f"[Check] There are {NP_TRAIN_TXT.shape[0]} pairs in train.txt.")
    print(f"[Check] There are {NP_TEST_TXT.shape[0]} pairs in test.txt.")
    print(f"[Check] There are {NP_VAL_TXT.shape[0]} pairs in val.txt.\n")
    
    return NP_TRAIN_TXT, NP_TEST_TXT, NP_VAL_TXT, len(NP_TRAIN_TXT), len(NP_TEST_TXT), len(NP_VAL_TXT)

Root_Path = "C:/Users/USER/Desktop/Projects/Github_Repo/AI/DeepLearning/__HW1_DATA/"
train_meta, test_meta, val_meta, len_train_meta, len_test_meta, len_val_meta = read_metadata_files(Root_Path)
print(f"The shape of test_meta is {test_meta.shape}")
print(f"The path of 1st example in test_meta is {test_meta[0][0]}")
print(f"The label of 1st example in test_meta is {test_meta[0][1]}")


from tqdm import tqdm
import cv2 as cv
import numpy as np

#len_dataset = len_test_meta
#dataset = test_meta
#(height, width) = (28, 28)

def load_dataset(len_dataset, dataset, height, width):
    img_dataset = np.zeros((len_dataset, height, width, 3))
    img_label = np.zeros(len_dataset)

    for i in tqdm(range(len_dataset)):
        # 取 label
        img_label[i] = dataset[i][1]
        # 取 input
        img = cv.imread(Root_Path + dataset[i][0])
        img_resize = cv.resize(img, (height, width))
        
    
        # 把 img 放入 dataset
        img_dataset[i] = img_resize
    return img_dataset, img_label

# read the dataset with load func

train_image_2, train_label_2 = load_dataset(len_train_meta, train_meta, 28, 28)
test_image_2, test_label_2 = load_dataset(len_test_meta, test_meta, 28, 28)
val_image_2, val_label_2 = load_dataset(len_val_meta, val_meta, 28, 28)



n_m_2, n_m_test_2, n_m_val_2 = len(train_label_2), len(test_label_2), len(val_label_2)
print("The shape of training image:", train_image_2.shape)
print("The shape of testing image: ", test_image_2.shape)
print("The shape of val image: ", val_image_2.shape)

print("Length of the training set: ", n_m_2)
print("Length of the testing set: ", n_m_test_2)
print("Length of the val set: ", n_m_val_2)

print("Shape of a single image: ", train_image_2[0].shape)


# Zero Padding and Normalization
train_image_normalized_pad_2 = normalize(zero_pad(train_image_2[:,:,:,:], 2),'lenet5')
test_image_normalized_pad_2  = normalize(zero_pad(test_image_2[:,:,:,:],  2),'lenet5')
val_image_normalized_pad_2  = normalize(zero_pad(val_image_2[:,:,:,:],  2),'lenet5')
print("The shape of training image with padding:", train_image_normalized_pad_2.shape)
print("The shape of testing image with padding: ", test_image_normalized_pad_2.shape)
print("The shape of testing image with padding: ", val_image_normalized_pad_2.shape)


import matplotlib.pyplot as plt
import numpy as np

index = np.random.randint(len_train_meta)
plt.imshow(train_image_normalized_pad_2[index])
print(train_label[index])
plt.show()



# The fixed weight (7x12 preset ASCII bitmaps) used in the RBF layer.
import numpy as np
bitmap = np.random.normal(-1,1, size = (10,84))
#print(bitmap.shape)
#bitmap = rbf_init_weight()
#print(bitmap.shape)
fig, axarr = plt.subplots(2,5,figsize=(20,8))
for i in range(10):
    x,y = int(i/5), i%5
    axarr[x,y].set_title(str(i))
    axarr[x,y].imshow(bitmap[i,:].reshape(12,7), cmap=mpl.cm.Greys)