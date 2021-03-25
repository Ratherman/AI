###########################
# [1] Read Metadata Files #
###########################

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

##########################
# [2] Color Of Histogram #
##########################

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

##########################
# [3] Create COH Dataset #
##########################

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
#from google.colab.patches import cv2_imshow
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

####################
# [4] Save Dataset #
####################


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

#########################
# [5] Sigmoid & Softmax #
#########################

# [Input Vars]
#   1. <ndarray> Z

# [Output Vars]
#   1. <ndarray> A

import numpy as np

def Sigmoid(Z):
  A = 1/(1 + np.exp(-Z))
  return A


# [Input Vars]
#   1. <ndarray> A

# [Output Vars]
#   1. <ndarray> Y_pred

def Softmax(A):
  Y_pred = np.exp(A-np.max(A))/np.sum(np.exp(A-np.max(A)))
  return Y_pred

##########################
# [6] Initialize Weights #
##########################

# [Input Vars] None

# [Output Vars]
#   1. <ndarray> W1: Its shape should be (769, 300)
#   2. <ndarray> W2: Its shape should be (300, 50)

import numpy as np

def initialize_weights():
  np.random.seed(0)
  W1 = np.random.uniform(low=-0.01, high=0.01, size=(769,300))
  W2 = np.random.uniform(low=-0.01, high=0.01, size=(300,50))
  assert W1.shape == (769, 300), f"[Error] W1's shape is {W1.shape}. Expected shape is (769, 300)."
  assert W2.shape == (300, 50), f"[Error] W2's shape is {W2.shape}. Expected shape is (300, 50)."
  return W1, W2

###########################
# [7] Forward Propagation #
###########################

# [Input Vars]
#   1. <ndarray> X: Its shape should be (1, 769).
#   2. <ndarray> W1: Its shape should be (769, 300).
#   3. <ndarray> W2: Its shape should be (300, 50).

# [Output Vars] 
#   1. <ndarray> S: Its shape should be (1, 50).
#   2. <ndarray> a2: Its shape should be (1, 50).
#   3. <ndarray> a1: Its shape should be (1, 300).

import numpy as np

def forward_pass(X, W1, W2):

  assert X.shape == (1, 769), f"[Error] X's shape is {X.shape}. Expected shape is (1, 769)."
  assert W1.shape == (769, 300), f"[Error] W1's shape is {W1.shape}. Expected shape is (769, 300)."
  assert W2.shape == (300, 50), f"[Error] W2's shape is {W2.shape}. Expected shape is (300, 50)."

  Z1 = np.dot(X, W1)
  assert Z1.shape == (1, 300), f"[Error] Z1's shape is {Z1.shape}. Expected shape is (1, 300)."

  A1 = Sigmoid(Z1)
  assert A1.shape == (1, 300), f"[Error] A1's shape is {A1.shape}. Expected shape is (1, 300)."

  Z2 = np.dot(A1, W2)
  assert Z2.shape == (1, 50), f"[Error] Z2's shape is {Z2.shape}. Expected shape is (1, 50)."

  A2 = Sigmoid(Z2)
  assert A2.shape == (1, 50), f"[Error] A2's shape is {A2.shape}. Expected shape is (1, 50)."

  Y_pred = Softmax(A2)
  assert Y_pred.shape == (1, 50), f"[Error] Y_pred's shape is {S.shape}. Expected shape is (1, 50)."

  return Y_pred, A2, A1

#####################
# [8] Cross Entropy #
#####################

# [Input Vars]
#   1. <ndarray> Y_pred: Its shape should be (1, 50).
#   2. <ndarray> Y_truth: Its shape should be (1, 50).

# [Output Vars]
#   2. <ndarray> Error

import numpy as np

def CrossEntropy(Y_pred, Y_truth):
  assert Y_truth.shape == (1, 50), f"[Error] Y_truth's shape is {Y_truth.shape}. Expected shape is (1, 50)."
  assert Y_pred.shape == (1, 50), f"[Error] Y_pred's shape is {S.shape}. Expected shape is (1, 50)."

  Error = (-1 * Y_truth * np.log(Y_pred)).sum()
  return Error

############################
# [9] Backward Propagation #
############################

# [Input Vars]
#   1. <ndarray> Y_pred: Its shape should be (1, 50).
#   2. <ndarray> Y_truth: Its shape should be (1, 50).
#   3. <ndarray> A2: Its shape should be (1, 50).
#   4. <ndarray> A1: Its shape should be (1, 300).
#   5. <ndarray> X: Its shape should be (1, 769).
#   6. <ndarray> W2: Its shape should be (300, 50).
#   7. <ndarray> W1: Its shape should be (769, 300).

# [Output Vars]
#   1. <ndarray> dEdW1: Its shape should be the same as W1, which is (769, 300).
#   2. <ndarray> dEdW2: Its shape should be the same as W2, which is (300, 50).

import numpy as np

def backward_pass(Y_pred, Y_truth, A2, A1, X, W2, W1):
  assert Y_pred.shape == (1, 50), f"[Error] Y_pred's shape is {Y_pred.shape}. Expected shape is (1, 50)."
  assert Y_truth.shape == (1, 50), f"[Error] Y_truth's shape is {Y_truth.shape}. Expected shape is (1, 50)."
  assert A2.shape == (1, 50), f"[Error] A2's shape is {A2.shape}. Expected shape is (1, 50)."
  assert A1.shape == (1, 300), f"[Error] A1's shape is {A1.shape}. Expected shape is (1, 300)."
  assert X.shape == (1, 769), f"[Error] X's shape is {X.shape}. Expected shape is (1, 769)."

  dEdA2 = Y_pred - Y_truth
  assert dEdA2.shape == (1, 50), f"[Error] dEdA2's shape is {dEdA2.shape}. Expected shape is (1, 50)."
  
  dZ2_local = np.multiply(1 - A2, A2)
  dEdZ2 = np.multiply(dZ2_local, dEdA2)
  assert dEdZ2.shape == (1, 50), f"[Error] dEdZ2's shape is {dEdZ2.shape}. Expected shape is (1, 50)."

  dEdW2 = np.outer(A1, dEdZ2)
  assert dEdW2.shape == (300, 50), f"[Error] dEdW2's shape is {dEdW2.shape}. Expected shape is (300, 50)."

  dEdA1 = np.dot(dEdZ2, W2.T)
  assert dEdA1.shape == (1, 300), f"[Error] dEdA1's shape is {dEdA1.shape}. Expected shape is (1, 300)."

  dZ1_local = np.multiply(1 - A1, A1)
  dEdZ1 = np.multiply(dZ1_local, dEdA1)
  assert dEdZ1.shape == (1, 300), f"[Error] dEdZ1's shape is {dEdZ1.shape}. Expected shape is (1, 300)."

  dEdW1 = np.outer(X, dEdZ1)
  assert dEdW1.shape == (769, 300), f"[Error] dEdW1's shape is {dEdW1.shape}. Expected shape is (769, 300)."

  dEdX = np.dot(dEdZ1, W1.T)
  assert dEdX.shape == (1, 769), f"[Error] dEdX.shape's shape is {dEdX.shape}. Expected shape is (1, 769)."

  return dEdW1, dEdW2

#######################
# [10] Update Weights #
#######################

# [Input Vars]
#   1. <ndarray> dEdW1
#   2. <ndarray> dEdW2
#   3. <ndarray> W1
#   4. <ndarray> W2
#   5. <float> lr

# [Output Vars]
#   1. <ndarray>
#   2. <ndarray>

import numpy as np

def update_weights(dEdW1, dEdW2, W1, W2, lr):
  W1 = W1 - lr * dEdW1
  W2 = W2 - lr * dEdW2
  return W1, W2

####################################
# [11] Provide Empty Accuracy List #
####################################

# [Input Vars] None

# [Output Vars]
#   1. <list> Trian_Accuracy_top_1
#   2. <list> Val_Accuracy_top_1
#   3. <list> Train_Accuracy_top_5
#   4. <list> Val_Accuracy_top_5

def provide_empty_accuracy_list():

  # Accuracy_top_1 will contain the top-1 accuracy per epoch
  Trian_Accuracy_top_1 = []
  Val_Accuracy_top_1 = []

  # Accuracy_top_5 will contain the top-5 accuracy per epoch
  Train_Accuracy_top_5 = []
  Val_Accuracy_top_5 = []
  
  return Trian_Accuracy_top_1, Val_Accuracy_top_1, Train_Accuracy_top_5, Val_Accuracy_top_5

#####################
# [12] Top Accuracy #
#####################

# [Input Vars]
#   1. <list> Dataset: Either Train_COH_Dataset, Val_COH_Dataset, or Test_COH_Dataset
#   2. <list> Label: Train_COH_Label, Val_COH_Label, Test_COH_Label
#   3. <ndarry> W: It's the updated Weight from training process.
#   4. <float> Scale: It's the hyper parameter we decided in training process.
#   5. <String> Name: It's for convenient purpose

# [Output Vars]
#   1. <int> top1_accuracy
#   2. <int> top5_accuracy

def top_accuracy(Dataset, Label, W1, W2, Scale, Name):

  num_top1_pred = 0
  num_top5_pred = 0
  len_dataset = len(Label)

  for i in range(len_dataset):
    
    # 1. Grab the i-th data
    X = np.array(Dataset[i:i+1]).reshape(1, 769)/Scale
    Y = int(Label[i:i+1][0])
    assert X.shape == (1,769), f"[Error] X's shape is {X.shape}. Expected shape is (1, 769)."
    
    # 3. Predict the label by using Softmax.
    Y_pred, A1, A2 = forward_pass(X, W1, W2)
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

#######################
# [13] Top 5 Accuracy #
#######################

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

##################################################
# Prepare Training, Testing & Validation Dataset #
##################################################

#PATH_TO_DESIRED_LOCATION = "gdrive/MyDrive/AI_Dataset/"
#np_train_txt, np_test_txt, np_val_txt = read_metadata_files(PATH_TO_DESIRED_LOCATION)

#Val_COH_Dataset, Val_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_val_txt, 128)
#save_dataset(Val_COH_Dataset, Val_COH_Label, "Val_COH_Dataset.csv", "Val_COH_Label.csv")

#Test_COH_Dataset, Test_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_test_txt, 128)
#save_dataset(Test_COH_Dataset, Test_COH_Label, "Test_COH_Dataset.csv", "Test_COH_Label.csv")

#Train_COH_Dataset, Train_COH_Label = create_COH_dataset(PATH_TO_DESIRED_LOCATION, np_train_txt, 128)
#save_dataset(Train_COH_Dataset, Train_COH_Label, "Train_COH_Dataset.csv", "Train_COH_Label.csv")


####################
# Read COH Dataset #
####################

import pandas as pd

ROOT_PATH = "C:/Users/USER/Desktop/Projects/Github_Repo/AI/DeepLearning/__HW1_DATA/"
train_X = pd.read_csv(ROOT_PATH + "Train_COH_Dataset.csv", header=None)
train_Y = pd.read_csv(ROOT_PATH + "Train_COH_Label.csv", header=None)

val_X = pd.read_csv(ROOT_PATH + "Val_COH_Dataset.csv", header=None)
val_Y = pd.read_csv(ROOT_PATH + "Val_COH_Label.csv", header=None)

###############################################
# Build Multi-Class-Classification Perceptron #
###############################################
import numpy as np
import time

# Initialize weight matrix W
np.random.seed(0)
W1 = np.random.uniform(low=-0.01, high=0.01, size=(769,300))
W2 = np.random.uniform(low=-0.01, high=0.01, size=(300,50))
### W1, W2 = initialize_weights()

# Setup hyper parameters
Epoch = 100
lr = 0.03
Scale = 1000.0

# Accuracy_top_1 will contain the top-1 accuracy per epoch
Trian_Accuracy_top_1 = []
Val_Accuracy_top_1 = []

# Accuracy_top_5 will contain the top-5 accuracy per epoch
Train_Accuracy_top_5 = []
Val_Accuracy_top_5 = []

### Trian_Accuracy_top_1, Val_Accuracy_top_1, Train_Accuracy_top_5, Val_Accuracy_top_5 = provide_empty_accuracy_list()

# E will contain the average cross-entropy per epoch
E = []
len_COH_Dataset = len(train_X)
tic = time.time()
for epoch in range(Epoch):
  # shuffle the training dataset.
  random_index = np.arange(len_COH_Dataset)
  np.random.shuffle(random_index)

  # e will record errors in an epoch.
  e = []
  for i in random_index:
    # Grab the i-th training data.
    X = np.array(train_X[i:i+1]).reshape(1, 769)/Scale

    # Forward Pass
    Y_pred, A2, A1 = forward_pass(X, W1, W2)

    # Grab the i-th label of i-th training data.
    label = int(train_Y[i:i+1][0])
    Y_truth = np.zeros(50).reshape(1,50)
    Y_truth[0][label] = 1
    assert Y_truth.shape == (1,50), f"[Error] Y_truth's shape is {Y_truth.shape}. Expected shape is (1, 50)."
    
    # Record cross entropy error
    e.append(round(CrossEntropy(Y_pred, Y_truth),4))

    # Backward Pass
    dEdW1, dEdW2 = backward_pass(Y_pred, Y_truth, A2, A1, X, W2, W1)

    # Update the parameters
    W1, W2 = update_weights(dEdW1, dEdW2, W1, W2, lr)

  toc = time.time()
  print(f"\n[Training] Epoch: {epoch}, the Cross Entropy Loss is {round(np.mean(e),4)}. In this epoch, I spent {round(toc - tic,2)} sec.")
  tic = time.time()

  # Measure the top-1 accuracy and top-5 accuracy
  train_top1_accuracy, train_top5_accuracy = top_accuracy(train_X, train_Y, W1, W2, Scale, "Train")
  val_top1_accuracy, val_top5_accuracy = top_accuracy(val_X, val_Y, W1, W2, Scale, "Val")

  # Collect results
  E.append(np.mean(e))
  Trian_Accuracy_top_1.append(train_top1_accuracy)
  Train_Accuracy_top_5.append(train_top5_accuracy)
  Val_Accuracy_top_1.append(val_top1_accuracy)
  Val_Accuracy_top_5.append(val_top5_accuracy)

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
plt.show()

