import pandas as pd

root_path = "C:/Users/AC/Desktop/Github/AI/DeepLearning/HW1/"
train_X = pd.read_csv(root_path + "Train_COH_Dataset_all_256.csv", header=None)
train_Y = pd.read_csv(root_path + "Train_COH_Label_all_256.csv", header=None)
train_Y = train_Y[0].values
train_Y = train_Y.astype(int)

val_X = pd.read_csv(root_path + "Val_COH_Dataset_RF.csv", header=None)
val_Y = pd.read_csv(root_path + "Val_COH_Label_RF.csv", header=None)
val_Y = val_Y[0].values
val_Y = val_Y.astype(int)

test_X = pd.read_csv(root_path + "Test_COH_Dataset_RF.csv", header=None)
test_Y = pd.read_csv(root_path + "Test_COH_Label_RF.csv", header=None)
test_Y = test_Y[0].values
test_Y = test_Y.astype(int)

import time
import xgboost as xgb

dtrain = xgb.DMatrix(train_X, label=train_Y)
dval = xgb.DMatrix(val_X, label=val_Y)
dtest = xgb.DMatrix(test_X, label=test_Y)

param = {
    "max_depth":3, # the maximum number of teach tree
    "eta":0.01, # the training step for each iteration
    "silent":1, # loggin mode - quiet
    "objective": "multi:softprob", # error evaluation for multiclass training
    "num_class":50 # the number of classes exists in this datasets
}

num_round = 30
tic = time.time()
bst = xgb.train(param, dtrain, num_round)
toc = time.time()

print(f"[Analysis] I have spent {round(toc - tic, 2)} sec to train the XGboost model.")

preds_val = bst.predict(dval)
preds_test = bst.predict(dtest)

import numpy as np
best_pred_val = np.asarray([np.argmax(line) for line in preds_val])
best_pred_test = np.asarray([np.argmax(line) for line in preds_test])

from sklearn import metrics
print(f"[Result of Val] Accuracy = {round(metrics.accuracy_score(val_Y, best_pred_val) * 100, 2)} %.")
print(f"[Result of Test] Accuracy = {round(metrics.accuracy_score(test_Y, best_pred_test) * 100, 2)} %.")