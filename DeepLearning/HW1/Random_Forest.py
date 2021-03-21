import pandas as pd

root_path = "C:/Users/USER/Desktop/Projects/Github_Repo/AI/DeepLearning/"
train_X = pd.read_csv(root_path + "Train_COH_Dataset.csv", header=None)
train_Y = pd.read_csv(root_path + "Train_COH_Label.csv", header=None)
train_Y = train_Y[0].values
train_Y = train_Y.astype(int)

val_X = pd.read_csv(root_path + "Val_COH_Dataset.csv", header=None)
val_Y = pd.read_csv(root_path + "Val_COH_Label.csv", header=None)
val_Y = val_Y[0].values
val_Y = val_Y.astype(int)

test_X = pd.read_csv(root_path + "Test_COH_Dataset.csv", header=None)
test_Y = pd.read_csv(root_path + "Test_COH_Label.csv", header=None)
test_Y = test_Y[0].values
test_Y = test_Y.astype(int)

import time
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 30)

tic = time.time()
rf_model.fit(train_X, train_Y)
toc = time.time()
print(f"[Analysis] I have spent {round(toc - tic, 2)} sec to train the RF model.")

pred_val = rf_model.predict(val_X)
pred_test = rf_model.predict(test_X)

from sklearn import metrics
print(f"[Result of Val] Accuracy = {round(metrics.accuracy_score(val_Y, pred_val) * 100, 2)} %.")
print(f"[Result of Test] Accuracy = {round(metrics.accuracy_score(test_Y, pred_test) * 100, 2)} %.")