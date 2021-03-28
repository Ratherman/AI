from nn import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Read Dataset
# 2. Build & Train NN (Set Hyper Parameters here.)
# 3. Draw Top-1/5 Accuracy and Cross Entropy


###################
# 1. Read Dataset #
###################

ROOT_PATH = "C:/Users/USER/Desktop/Projects/Github_Repo/AI/DeepLearning/__HW1_DATA/"

# Training Dataset
train_X = pd.read_csv(ROOT_PATH + "Train_COH_Dataset.csv", header=None)
train_Y = pd.read_csv(ROOT_PATH + "Train_COH_Label.csv", header=None)
len_train_X = len(train_X)

# Validation Dataset
val_X = pd.read_csv(ROOT_PATH + "Val_COH_Dataset.csv", header=None)
val_Y = pd.read_csv(ROOT_PATH + "Val_COH_Label.csv", header=None)

# Testing Dataset
test_X = pd.read_csv(ROOT_PATH + "Test_COH_Dataset.csv", header=None)
test_Y = pd.read_csv(ROOT_PATH + "Test_COH_Label.csv", header=None)

#######################
# 2. Build & Train NN #
#######################

# Initialize the Custom Class
NN = nn()

# Initialize Weight Matrix
W1, W2 = NN.initialize_weights()

# Accuracy_top_1 and Accuracy_top_5 will record the top-1 and top-5 accuracies. And E will record the loss.
Trian_Accuracy_top_1, Val_Accuracy_top_1, Train_Accuracy_top_5, Val_Accuracy_top_5, E = [], [], [], [], []

############ Hyper Parameter ############
Epoch = 100
lr_1, lr_2 = 0.01, 0.01
Scale = 100.0
############ Hyper Parameter ############

tic = time.time()
for epoch in range(Epoch):

  # Shuffle the training dataset.
  random_index = np.arange(len_train_X)
  np.random.shuffle(random_index)

  e = []
  for i in random_index:
    # Grab the i-th training data.
    X = np.array(train_X[i:i+1]).reshape(1, 769)/Scale

    # Forward Propagation.
    Y_pred, A2, A1 = NN.forward_pass(X, W1, W2)

    # Grab the i-th label.
    label = int(train_Y[i:i+1][0])
    Y_truth = np.zeros(50).reshape(1,50)
    Y_truth[0][label] = 1
    assert Y_truth.shape == (1,50), f"[Error] Y_truth's shape is {Y_truth.shape}. Expected shape is (1, 50)."
    
    # Record cross entropy.
    e.append(round(NN.cross_entropy(Y_pred, Y_truth),4))

    # Backward Propagation.
    dEdW1, dEdW2 = NN.backward_pass(Y_pred, Y_truth, A2, A1, X, W2, W1)

    # Update Weight Matrix.
    if (epoch < 50):
      W1, W2 = NN.update_weights(dEdW1, dEdW2, W1, W2, lr_1)
    else:
      W1, W2 = NN.update_weights(dEdW1, dEdW2, W1, W2, lr_2)

  toc = time.time()
  print(f"\n[Training] Epoch: {epoch}, the Cross Entropy Loss is {round(np.mean(e),4)}. In this epoch, I spent {round(toc - tic,2)} sec.")
  tic = time.time()

  # Measure the top-1 accuracy and top-5 accuracy
  train_top1_accuracy, train_top5_accuracy = NN.top_accuracy(train_X, train_Y, W1, W2, Scale, "Train")
  val_top1_accuracy, val_top5_accuracy = NN.top_accuracy(val_X, val_Y, W1, W2, Scale, "Val")

  # Collect results
  E.append(np.mean(e))
  Trian_Accuracy_top_1.append(train_top1_accuracy)
  Train_Accuracy_top_5.append(train_top5_accuracy)
  Val_Accuracy_top_1.append(val_top1_accuracy)
  Val_Accuracy_top_5.append(val_top5_accuracy)

test_top1_accuracy, test_top5_accuracy = NN.top_accuracy(test_X, test_Y, W1, W2, Scale, "Test")

##############################################
# 3. Draw Top-1/5 Accuracy and Cross Entropy #
##############################################

# Draw Diagram: Accuracy
plt.figure(figsize=(20,10))
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Accuracy %", fontsize=20)

plt.plot(Trian_Accuracy_top_1, label="Train Top-1")
plt.plot(Train_Accuracy_top_5, label="Train Top-5")
plt.plot(Val_Accuracy_top_1, label="Val Top-1")
plt.plot(Val_Accuracy_top_5, label="Val Top-5")
plt.legend(loc=2, fontsize=20)
plt.show()

# Draw Diagram: Cross Entropy
plt.figure(figsize=(20,10))
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Loss: Cross Entropy", fontsize=20)

plt.plot(E)
plt.legend(loc=2, fontsize=20)
plt.show()