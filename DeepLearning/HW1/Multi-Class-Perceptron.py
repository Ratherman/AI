# [Input Vars]
#   1. <string> PATH_TO_DESIRED_LOCATION: It should be the directory containing (1) images/ (2) train.txt (3) test.txt (4) val.txt

# [Output Vars]
#   1. <ndarray> np_train_txt: It contains both the directory to a specific image and the related label.
#   2. <ndarray> np_test_txt: It contains both the directory to a specific image and the related label.
#   3. <ndarray> np_val_txt: It contains both the directory to a specific image and the related label.
import pandas as pd
import numpy as np
def read_metadata_files(PATH_TO_DESIRED_LOCATION):

  # test.txt
  test_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION+"test.txt", sep=" ")
  np_test_txt = np.array(test_txt)

  # train.txt
  train_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION+"train.txt", sep=" ")
  np_train_txt = np.array(train_txt)

  # val.txt
  val_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION+"val.txt", sep=" ")
  np_val_txt = np.array(val_txt)

  print(f"[Check] There are {np_train_txt.shape[0]} pairs in train.txt.")
  print(f"[Check] There are {np_test_txt.shape[0]} pairs in test.txt.")
  print(f"[Check] There are {np_val_txt.shape[0]} pairs in val.txt.\n")

  return np_train_txt, np_test_txt, np_val_txt


# [Input Vars]
#   1. <ndarray> img: expected a square image.

# [Output Vars]
#   1. <list> X: a list contains 769 elements.
from collections import Counter
def color_of_histogram(img):

  b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
  channels = [b,g,r]

  # Add bias term for perceptron.
  X = [1]    
  
  for c in channels:
    cnt = Counter(c.reshape(img.shape[0]*img.shape[0]))
    
    for key in range(256):
      if key not in cnt.keys(): 
        X.append(0)
      else: 
        X.append(cnt[key])

  return X


# [Input Vars]
#   1. <string> metadata_file: train.txt, val.txt, test.txt
#   2. <int> image_size: each image has different size originally, so I convert each image to a fix size.

# [Output Vars]
#   1. <list> COH_Dataset: the data in COH_Dataset are a 1-D array containing 769 elements. 256*3+1=769
#   2. <list> COH_Label: the label of each data.

# For each image, we do the following things:
#   1. Read a specific image in RGB format, and also read the label.
#   2. Resize the image to a fixed size. (ex: 256,256)
#   3. Represent the image in form of Color of Histogram. (Also add a bias term at this step)
#   4. Append the 1-D array into the COH_Dataset
#   5. Append the label into the COH_Label
import cv2 as cv
import time
def create_COH_dataset(PATH_TO_DESIRED_LOCATION, metadata_file, image_size):
  COH_Dataset = []
  COH_Label = []
  counter = 0

  tic = time.time()
  for one_pair in metadata_file:
    # 1. Read a specific image in RGB format.
    img = cv.imread(PATH_TO_DESIRED_LOCATION + one_pair[0])
    img_label = one_pair[1]

    # 2. Resize the image to a fixed size.
    img_resize = cv.resize(img, (image_size,image_size))

    # 3. Represent the image in form of Color of Histogram.
    COH_array = color_of_histogram(img_resize)

    # 4. Append the 1-D array into the COH_Dataset.
    COH_Dataset.append(COH_array)

    # 5. Append the label into the COH_Label.
    COH_Label.append(img_label)

    # 6. Get to know the progress.
    counter = counter + 1
    if counter % 1000 == 0: 
      toc = time.time()
      print(f"[Note] I have finished transforming {counter} images into other space. This time, I spent {round(toc - tic,2)} sec.")
      tic = time.time()
  return COH_Dataset, COH_Label

# [Input Vars]
#   1. <list> Dataset: Train_COH_Dataset, Val_COH_Dataset, Test_COH_Dataset
#   2. <string> desired_name: 

# [Outpu Vars] None
from numpy import asarray
from numpy import savetxt
import time
def save_dataset(Dataset, Label, dataset_name, label_name):
  tic = time.time()
  dataset = asarray(Dataset)
  savetxt(dataset_name, dataset, delimiter=',')

  label = asarray(Label)
  savetxt(label_name, label, delimiter=',')
  toc = time.time()

  print(f"[Save Success] I have spent {round(toc-tic,2)} sec to save the {dataset_name} and the {label_name}.")

import numpy as np
def CrossEntropy(y_pred, y_truth):
  return (-1*y_truth*np.log(y_pred)).sum()
def Softmax(Z):
  S = np.exp(Z-np.max(Z))/np.sum(np.exp(Z-np.max(Z)))
  return S

# [Input Vars]
#   1. <list> Dataset: Either Train_COH_Dataset, Val_COH_Dataset, or Test_COH_Dataset
#   2. <list> Label: Train_COH_Label, Val_COH_Label, Test_COH_Label
#   3. <ndarry> W: It's the updated Weight from training process.
#   4. <float> Scale: It's the hyper parameter we decided in training process.
#   5. <String> Name: It's for convenient purpose

# [Output Vars]
#   1. <int> top1_accuracy
#   2. <int> top5_accuracy
def top_accuracy(Dataset, Label, W, Scale, Name):

  num_top1_pred = 0
  num_top5_pred = 0
  len_dataset = len(Label)

  for i in range(len_dataset):
    
    # 1. Grab the i-th data
    X = np.array(Dataset[i]).reshape(1, 769)/Scale
    Y = Label[i]
    assert X.shape == (1,769), f"[Error] X's shape is {X.shape}. Expected shape is (1, 769)."
    
    # 2. Compute Z by using X and W.
    Z = np.dot(X, W)
    assert W.shape == (769,50), f"[Error] W's shape is {W.shape}. Expected shape is (769, 50)."
    assert Z.shape == (1,50), f"[Error] Z's shape is {X.shape}. Expected shape is (1, 50)."

    # 3. Predict the label by using Softmax.
    Y_pred = Softmax(Z)
    assert Y_pred.shape == (1,50) , f"[Error] Y_pred's shape is {Y_pred.shape}. Expected shape is (1, 50)."

    # 4. Grab top 5 predictions.
    top_1, top_2, top_3, top_4, top_5 = grab_top_5_predictions(Y_pred)

    # 5. Check if the label is the top 1 prediction.
    if Y == top_1: num_top1_pred = num_top1_pred + 1

    # 6. Check if the label is in the top 5 predictions
    if Y in [top_1, top_2, top_3, top_4, top_5]: num_top5_pred = num_top5_pred + 1
  
  top1_accuracy = round(num_top1_pred/len_dataset*100, 2)
  top5_accuracy = round(num_top5_pred/len_dataset*100, 2)
  print(f"[Result of {Name}] The top-1 accuracy is {top1_accuracy} %")
  print(f"[Result of {Name}] The top-5 accuracy is {top5_accuracy} %")
  return top1_accuracy, top5_accuracy

# [Input Vars]
#   1. <ndarray> Y_pred: It's a 1-D ndarray which contains the possibilities of the predictions.

# [Output Vars]
#   1. <int> top_1: The 1st likely breed among those 50 breeds.
#   2. <int> top_2: The 2nd likely breed among those 50 breeds.
#   3. <int> top_3: The 3rd likely breed among those 50 breeds.
#   4. <int> top_4: The 4th likely breed among those 50 breeds.
#   5. <int> top_5: The 5th likely breed among those 50 breeds.
import numpy as np
def grab_top_5_predictions(Y_pred):

    top_1 = Y_pred.argmax()
    Y_pred[0][top_1] = 0

    top_2 = Y_pred.argmax()
    Y_pred[0][top_2] = 0
    
    top_3 = Y_pred.argmax()
    Y_pred[0][top_3] = 0

    top_4 = Y_pred.argmax()
    Y_pred[0][top_4] = 0

    top_5 = Y_pred.argmax()

    return top_1, top_2, top_3, top_4, top_5

#########################
# Create & Save Dataset #
#########################

PATH_TO_DESIRED_LOCATION = "C:/Users/USER/Desktop/Projects/Github_Repo/AI/DeepLearning/"
np_train_txt, np_test_txt, np_val_txt = read_metadata_files(PATH_TO_DESIRED_LOCATION)

# Training Dataset
Train_COH_Dataset, Train_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_train_txt, 256)
save_dataset(Train_COH_Dataset, Train_COH_Label, "Train_COH_Dataset.csv", "Train_COH_Label.csv")

# Validation Dataset
Val_COH_Dataset, Val_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_val_txt, 256)
save_dataset(Val_COH_Dataset, Val_COH_Label, "Val_COH_Dataset.csv", "Val_COH_Label.csv")

# Testing Dataset
Test_COH_Dataset, Test_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_test_txt, 256)
save_dataset(Test_COH_Dataset, Test_COH_Label, "Test_COH_Dataset.csv", "Test_COH_Label.csv")

np_Train_COH_Dataset = np.array(Train_COH_Dataset)
np_Train_COH_Label = np.array(Train_COH_Label).reshape(len(Train_COH_Label),1)
np_Val_COH_Dataset = np.array(Val_COH_Dataset)
np_Val_COH_Label = np.array(Val_COH_Label).reshape(len(Val_COH_Label), 1)
print(f"The shape of np_Train_COH_Dataset is {np_Train_COH_Dataset.shape}, and the shape of np_Train_COH_Label is {np_Train_COH_Label.shape}")

###############################################
# Build Multi-Class-Classification Perceptron #
###############################################

import numpy as np
import time

# Initialize weight matrix W
np.random.seed(0)
W = np.random.uniform(low=-0.01, high=0.01, size=(769,50))
# Setup hyper parameters
Epoch = 50
r = 0.0003
Scale = 1000.0

# Accuracy_top_1 will contain the top-1 accuracy per epoch
Trian_Accuracy_top_1 = []
Val_Accuracy_top_1 = []

# Accuracy_top_5 will contain the top-5 accuracy per epoch
Train_Accuracy_top_5 = []
Val_Accuracy_top_5 = []

# E will contain the average cross-entropy per epoch
E = []
len_COH_Dataset = len(Train_COH_Label)
tic = time.time()
for epoch in range(Epoch):
  # shuffle the training dataset.
  random_index = np.arange(len_COH_Dataset)
  np.random.shuffle(random_index)

  # e will record errors in an epoch.
  e = []
  for i in random_index:
    # Grab the i-th training data.
    X = np.array(Train_COH_Dataset[i]).reshape(1, 769)/Scale
    assert X.shape == (1,769), f"[Error] X's shape is {X.shape}. Expected shape is (1, 769)."

    # Compute Z by using X and W.
    Z = np.dot(X, W)
    assert W.shape == (769,50), f"[Error] W's shape is {W.shape}. Expected shape is (769, 50)."
    assert Z.shape == (1,50), f"[Error] Z's shape is {X.shape}. Expected shape is (1, 50)."

    # Predict the label by using Softmax
    Y_pred = Softmax(Z)
    assert Y_pred.shape == (1,50) , f"[Error] Y_pred's shape is {Y_pred.shape}. Expected shape is (1, 50)."

    # Grab the i-th label of i-th training data.
    label = Train_COH_Label[i]
    Y_truth = np.zeros(50).reshape(1,50)
    Y_truth[0][label] = 1
    assert Y_truth.shape == (1,50), f"[Error] Y_truth's shape is {Y_truth.shape}. Expected shape is (1, 50)."
    
    # Record cross entropy error
    e.append(round(CrossEntropy(Y_pred, Y_truth),4))

    # Compute dE/dZ. This term is related to the derivative of softmax function. Magically the "dEdZ" is equal to "Y_pred - Y_truth"
    dEdZ = Y_pred - Y_truth
    assert dEdZ.shape == (1, 50), f"[Error] dEdZ's shape is {dEdZ.shape}. Expected shape is (1, 50)."

    # Compute the gradients w.r.t. the weight matrix W.
    dW = np.outer(X, dEdZ)
    assert dW.shape == (769, 50), f"[Error] dW's shape is {dW.shape}. Expected shape is (769, 50)."

    # Update the parameters
    W = W - r*dW

  toc = time.time()
  print(f"\n[Training] Epoch: {epoch}, the Cross Entropy Loss is {round(np.mean(e),4)}. In this epoch, I spent {round(toc - tic,2)} sec.")
  tic = time.time()

  # Measure the top-1 accuracy and top-5 accuracy
  train_top1_accuracy, train_top5_accuracy = top_accuracy(Train_COH_Dataset, Train_COH_Label, W, Scale, "Train")
  val_top1_accuracy, val_top5_accuracy = top_accuracy(Val_COH_Dataset, Val_COH_Label, W, Scale, "Val")
  test_top1_accuracy, test_top5_accuracy = top_accuracy(Test_COH_Dataset, Test_COH_Label, W, Scale, "Test")

  # Collect results
  E.append(np.mean(e))
  Trian_Accuracy_top_1.append(train_top1_accuracy)
  Train_Accuracy_top_5.append(train_top5_accuracy)
  Val_Accuracy_top_1.append(val_top1_accuracy)
  Val_Accuracy_top_5.append(val_top5_accuracy)

########################
# Evaluate the results #
########################

# [Input Vars]
#   1. <list> Accuracy_top_1
#   2. <list> Acciracy_top_5

# [Output Vars] None
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Accuracy %", fontsize=20)

plt.plot(Trian_Accuracy_top_1, label="Train Top-1")
plt.plot(Train_Accuracy_top_5, label="Train Top-5")
plt.plot(Val_Accuracy_top_1, label="Val Top-1")
plt.plot(Val_Accuracy_top_5, label="Val Top-5")
plt.legend(loc=2, fontsize=20)
#plt.savefig("Perceptron Accuracy: ImgSize_256 | Scale_1000 | Epoch_500 | Rate_10-4")
plt.show()

# [Input Vars]
#   1. <list> E

# [Output Vars] None
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Loss: Cross Entropy", fontsize=20)
plt.plot(E)
plt.legend(loc=2, fontsize=20)
#plt.savefig("Perceptron Loss: ImgSize_256 | Scale_1000 | Epoch_500 | Rate_10-4")
plt.show()

print("[Final]")
val_top1_accuracy, val_top5_accuracy = top_accuracy(Val_COH_Dataset, Val_COH_Label, W, Scale, "Val")
test_top1_accuracy, test_top5_accuracy = top_accuracy(Test_COH_Dataset, Test_COH_Label, W, Scale, "Test")