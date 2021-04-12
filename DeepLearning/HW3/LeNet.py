# [Input Vars]
#   1. <string> PATH_TO_DESIRED_LOCATION: It should be the directory containing (1) images/ (2) train.txt (3) test.txt (4) val.txt

# [Output Vars]
#   1. <ndarray> np_train_txt: It contains both the directory to a specific image and the related label.
#   2. <ndarray> np_test_txt: It contains both the directory to a specific image and the related label.
#   3. <ndarray> np_val_txt: It contains both the directory to a specific image and the related label.
import pandas as pd
import numpy as np
def read_metadata_files(PATH_TO_DESIRED_LOCATION):
    # train.txt
    train_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION+"train.txt", sep=" ")
    NP_TRAIN_TXT = np.array(train_txt)
    
    # test.txt
    test_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION+"test.txt", sep=" ")
    NP_TEST_TXT = np.array(test_txt)
    
    # val.txt
    val_txt = pd.read_csv(PATH_TO_DESIRED_LOCATION+"val.txt", sep=" ")
    NP_VAL_TXT = np.array(val_txt)
    
    print(f"[Check] There are {NP_TRAIN_TXT.shape[0]} pairs in train.txt.")
    print(f"[Check] There are {NP_TEST_TXT.shape[0]} pairs in test.txt.")
    print(f"[Check] There are {NP_VAL_TXT.shape[0]} pairs in val.txt.\n")
    
    return NP_TRAIN_TXT, NP_TEST_TXT, NP_VAL_TXT

################
# Zero Padding #
################

#[Input Vars]
#  1. <ndarray> X: Unpadded image. The shape is (n_H_prev, n_W_prev, n_C_prev).
#  2. <int> pad: expected number of pads on each side. The shape is (n_H_prev + 2 * pad, n_W_prev + 2 * pad, n_C_prev).
#
#[Output Vars]
#  1. <ndarray> X_pad: Padded image.

import numpy as np

def __zero_pad(X, pad):
    X_pad = np.pad(X, ((pad, pad), (pad, pad),(0,0)), "constant", constant_values = 0)
    return X_pad

####################
# Conv Single Step #
####################

#[Input Vars]
#  1. <ndarray> a_slice_prev: slice of previous feature maps. The shape is (f, f, n_C_prev).
#  2. <ndarray> K: A single weight matrix (kernel). The shape is (f, f, n_C_prev).
#  3. <ndarray> b: A single bias term. The shape is (1, 1, 1).
#
#[Output Vars]
#  1. <float> Z: a scalar derived from convolution operation.

import numpy as np

def __conv_single_step(s_slice, K, b):
    
    S = np.multiply(s_slice, K)
    Z = np.sum(S)
    Z = Z + float(b)
    
    return Z

############################
# Conv Forward Propagation #
############################

#[Input Vars]
#  1. <ndarray> S_prev: The previous feature maps (after activation and pooling). The shape is (n_H_prev, n_W_prev, n_C_prev).
#  2. <ndarray> K: Kernels in a layer. The shape is (f, f, n_C_prev, n_C).
#  3. <ndarray> b: biases in a layer. THe shape is (1, 1, 1, n_C).
#  4. <dictionary> hparam: this contains hyper parameters like "pad" and "stride".
#
#[Output Vars]
#  1. <ndarray> C: This would be the feature map in the next layer (but before activation). The shape is (n_H, n_W, n_C).
#  2. <dictionary> cache: Cache the values needed for backward propagation.

import numpy as np

def conv_forward(S_prev, K, b, hparam):
    
    # 1. Retrieve shape of A_prev. We need this to compute the shape of the feature map in the next layer.
    (n_H_prev, n_W_prev, n_C_prev) = S_prev.shape
    
    # 2. Retrieve shape of K. We also need this (i.e. f) to compute the shape of the feature map in the next layer.
    (f, f, n_C_prev, n_C) = K.shape
    
    # 3. Retrieve info. from hyper parameters. We need them to compute the shape of the feature map in the next layer, too.
    stride = hparam["stride"]
    pad = hparam["pad"]
    
    # 4. With info from 1. ~ 3., we can compute the dimension for the feature map in the next layer.
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    # 5. Initialize feature maps in the next layer with zeros. Note #Kernel is equal to #Channel of the feature map.
    C = np.zeros((n_H, n_W, n_C))
    
    # 6. Pad S_prev
    S_prev_pad = __zero_pad(S_prev, pad)
    
    # 7. Do Cross-Relation Operation. Note the shape of the output feature map would be (n_H, n_W, n_C).
    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                
                # Define the corners in the S_prev_pad.
                vert_head = h * stride
                vert_tail = vert_head + f
                hori_head = w * stride
                hori_tail = hori_head + f
                
                # Get the slice.
                S_prev_slice = S_prev_pad[vert_head:vert_tail, hori_head:hori_tail, :]
                
                # Feed it into __conv_single_step(a_slice, K, b). Note we use one kernel and one bias term at once.
                C[h, w, c] = __conv_single_step(S_prev_slice, K[:,:,:,c], b[:,:,:,c])
    
    # 8. Check if the output feature map have the valid shape.
    assert(C.shape == (n_H, n_W, n_C))
    
    # 9. Store the cache for backward propagation
    cache = (S_prev, K, b, hparam)
    
    return C, cache

############################
# Pool Forward Propagation #
############################

#[Input Vars]
#  1. <ndarray> A_prev: The previous feature maps (after activation). The shape is (n_H_prev, n_W_prev, n_C_prev).
#  2. <dictionary> hparam: It contains "f" and "stride".
#  3. <string> mode: Switch between "maxpooling" and "avgpooling". The shape is (n_H, n_W, n_C). (n_C = n_C_prev)
#
#[Output Vars]
#  1. <ndarray> S: The output feature map after pooling operation.

import numpy as np

def pool_forward(A_prev, hparam, mode = "maxpooling"):
    # 1. Retrieve shape of A_prev.
    (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 2. Retrieve info from hyper parameter
    f = hparam["f"]
    stride = hparam["stride"]

    # 3. Define the shape of output of pooling operation.
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # 4. Initialize the output feature map after pooling operation with zeros.
    S = np.zeros((n_H, n_W, n_C))
    
    # 5. Do Pooling Operation
    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                
                # Define the corners in the A_prev_pad.
                vert_head = h * stride
                vert_tail = vert_head + f
                hori_head = w * stride
                hori_tail = hori_head + f
                
                # Get the slice. (Note that there's only one channel involved. Not like conv_forward)
                A_prev_slice = A_prev[vert_head:vert_tail, hori_head:hori_tail, c]
                
                # Pooling operation
                if mode == "maxpooling":
                    S[h, w, c] = np.max(A_prev_slice)
                elif mode == "avgpooling":
                    S[h, w, c] = np.mean(A_prev_slice)
                    
    # 6. Check if the output feature map have the valid shape.
    assert(S.shape == (n_H, n_W, n_C))
    
    # 7. Store the cache for backward propagation
    cache = (A_prev, hparam)
    
    return S, cache

#############################
# Conv Backward Propagation #
#############################

#[Input Vars]
#  1. <ndarray> dC: gradient of the cost with respect to the output of the conv layer (C). The shape is (n_H, n_W, n_C).
#  2. <dictionary> cache: Cache of output of conv_forward()
#
#[Output Vars]
#  1. <ndarray> dS_prev: gradient of the cost w.r.t. the input of the conv layer (S). The shape is (n_H_prev, n_W_prev, n_C_prev).
#  2. <ndarray> dK: gradient of the cost w.r.t. the weights of the conv layer (K). The shape is (f, f, n_C_prev, n_C).
#  3. <ndarray> db: gradient of the cost w.r.t. the biases of the conv layer (b). The shape is (1, 1, 1, n_C).

def conv_backward(dC, cache):
    
    # 1. Retrieve info. from cache.
    (S_prev, K, b, hparam) = cache
    
    # 2. Retrieve the shape of S_prev.
    (n_H_prev, n_W_prev, n_C_prev) = S_prev.shape
    
    # 3. Retrieve the shape of Kernel.
    (f, f, n_C_prev, n_C) = K.shape
    
    # 4. Retieve info. from hyper parameters.
    stride = hparam["stride"]
    pad = hparam["pad"]
    
    # 5. Retrieve the shape of dC
    (n_H, n_W, n_C) = dC.shape
    
    # 6. Initialize dS_prev, dK, db with the correct shapes.
    dS_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))
    dK = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    # 7. Pad dS_prev and S_prev
    S_prev_pad = __zero_pad(S_prev, pad)
    dS_prev_pad = __zero_pad(dS_prev, pad)
    
    # 8. Do backward pass operation
    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                                
                # Define the corners in the A_prev_pad.
                vert_head = h * stride
                vert_tail = vert_head + f
                hori_head = w * stride
                hori_tail = hori_head + f
                    
                # Get the slice.
                S_prev_slice = S_prev_pad[vert_head:vert_tail, hori_head:hori_tail, :]
                
                # Update Gradients (dS_prev, dK, db) for the window
                dS_prev_pad[vert_head:vert_tail, hori_head:hori_tail, :] += K[:,:,:,c] * dC[h, w, c]
                dK[: , :, :, c] += S_prev_slice * dC[h, w, c]
                db[: , :, :, c] += dC[h, w, c]
                
    # 9. Unpad dS_prev_pad
    if (pad == 0):
        dS_prev = dS_prev_pad
    else:
        dS_prev[:, :, :] = dS_prev_pad[pad:-pad, pad:-pad, :]
    
    
    # 10 Check the validity of the shape
    assert (dS_prev.shape == (n_H_prev, n_W_prev, n_C_prev))
    
    return dS_prev, dK, db

############################
# Max Pool Backward helper #
############################

import numpy as np

def __create_mask_from_window(s):
    mask = (s == np.max(s))
    return mask

############################
# Avg Pool Backward helper #
############################

def __distribute_value(ds, shape):
    
    # 1. Retrieve dimensions from shape
    (n_H, n_W) = shape
    
    # 2. Compute the value to distribute on the matrix
    average = ds / (n_H * n_W)
    
    # 3. Create a matrix where each entry is the avg. value.
    a = np.ones(shape) * average
    return a

#############################
# Pool Backward Propagation #
#############################

#[Input Vars]
#  1. <ndarray> dS: gradient of cost w.r.t. the output of the pooling layer. The shape is the same as the shape of S.
#  2. <dictionary> cache: It contaions the output from the forward pass.
#  3. <string> mode: Switch between "maxpooling" and "avgpooling".
#
#[Output Vars]
#  1. <ndarray> dA_prev: gradient of cost w.r.t. the input of the pooling layer. The shape is the same as the shape of A_prev.

import numpy as np

def pool_backward(dS, cache, mode = "maxpooling"):
    
    # 1. Retrieve info. from cache
    (A_prev, hparam) = cache
    
    # 2. Retrieve hyper parameters
    stride = hparam["stride"]
    f = hparam["f"]
    
    # 3. Retrieve the shapes of A_prev and dS
    n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    n_H, n_W, n_C = dS.shape
    
    # 4. Initialize dA_prev with zeros.
    dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))
    
    # 5. Do Backward Pass Operation
    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                                
                # Define the corners in the A_prev_pad.
                vert_head = h * stride
                vert_tail = vert_head + f
                hori_head = w * stride
                hori_tail = hori_head + f
                
                # Compute the backward propagation in both modes
                if mode == "maxpooling":
                    # Use the corners and the specific "c" tp defome the current slice of A_prev
                    A_prev_slice = A_prev[vert_head:vert_tail, hori_head:hori_tail, c]
                    
                    # Create the mask from A_prev_slice
                    mask = __create_mask_from_window(A_prev_slice)
                    
                    # Update dA_prev
                    dA_prev[vert_head:vert_tail, hori_head:hori_tail, c] += np.multiply(mask, dS[h, w, c])
                elif mode == "avgpooling":
                    # Get the entry ds from dS
                    ds = dS[h, w, c]
                    
                    # Define the shape of the kernel as (f, f).
                    shape = (f, f)
                    
                    # Distribute it (ds) to the correct slice of dA_prev
                    dA_prev[vert_head:vert_tail, hori_head:hori_tail, c] += __distribute_value(ds, shape)
    
    # 6. Check the dA_prev has the valid shape 
    assert (dA_prev.shape == A_prev.shape)
    
    return dA_prev

########################################
# Actication Functions for Propagation #
########################################

# [Input Vars]
#   1. <ndarray> Z
#
# [Output Vars]
#   1. <ndarray> A

import numpy as np

def activation_forward(Z, mode):
    if mode == "sigmoid":
        A = 1/(1 + np.exp(-Z))
    elif mode == "relu":
        A = Z * (Z > 0)
    return A

def activation_backward(X, mode):
    if mode == "sigmoid":
        D_Z_local = np.multiply(1 - X, X)
    elif mode == "relu":
        D_Z_local = X
        D_Z_local[X<=0] = 0
        D_Z_local[X>0] = 1
    return D_Z_local

# [Input Vars]
#   1. <ndarray> A
#
# [Output Vars]
#   1. <ndarray> Y_pred
def __softmax(A):
    Y_pred = np.exp(A-np.max(A))/np.sum(np.exp(A-np.max(A)))
    return Y_pred

# Initiallize the Kernels, Biases, and hparams

def Initialize_Parameters(low, high):
    
    # C1
    K_C1 = np.random.uniform(low=low, high=high, size=(5, 5, 3, 6))
    b_C1 = np.random.uniform(low=low, high=high, size=(1, 1, 1, 6))
    hparam_C1 = {"stride": 1, "pad": 2}

    # S2
    hparam_S2 = {"f": 2, "stride": 2}

    # C3
    K_C3 = np.random.uniform(low=low, high=high, size=(5, 5, 6, 16))
    b_C3 = np.random.uniform(low=low, high=high, size=(1, 1, 1, 16))
    hparam_C3 = {"stride":1, "pad": 0}

    # S4
    hparam_S4 = {"f": 2, "stride": 2}

    # C5
    K_C5 = np.random.uniform(low=low, high=high, size=(5, 5, 16, 120))
    b_C5 = np.random.uniform(low=low, high=high, size=(1, 1, 1, 120))
    hparam_C5 = {"stride":1, "pad": 0}

    # W7
    W7 = np.random.uniform(low=low, high=high, size=(120, 84))

    # W8
    W8 = np.random.uniform(low=low, high=high, size=(84, 50))
    
    return K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8

################################
# LeNet5 - Forward Propagation #
################################

def LeNet5_forward(X, K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8, pool_mode = "avgpooling", act_mode = "sigmoid"):
    
    X_C1, cache_C1 = conv_forward(X, K_C1, b_C1, hparam_C1)
    X_A1 = activation_forward(X_C1, act_mode)
    X_S2, cache_S2 = pool_forward(X_A1, hparam_S2, pool_mode)
    X_C3, cache_C3 = conv_forward(X_S2, K_C3, b_C3, hparam_C3)
    X_A3 = activation_forward(X_C3, act_mode)
    X_S4, cache_S4 = pool_forward(X_A3, hparam_S4, pool_mode)
    X_C5, cache_C5 = conv_forward(X_S4, K_C5, b_C5, hparam_C5)
    X_A5 = activation_forward(X_C5, act_mode)
    X_A6 = X_A5.reshape(1, 120)
    X_Z7 = np.dot(X_A6, W7)
    #X_A7 = activation_forward(X_Z7, act_mode)
    #X_Z8 = np.dot(X_A7, W8)
    X_Z8 = np.dot(X_Z7, W8)
    X_A8 = activation_forward(X_Z8, act_mode)

    Y_pred = __softmax(X_A8)
    
    return cache_C1, X_A1, cache_S2, cache_C3, X_A3, cache_S4, cache_C5, X_A5, X_A6, X_A7, X_A8, Y_pred

def cross_entropy(Y_pred, Y_truth):
    Error = (-1 * Y_truth * np.log(Y_pred)).sum()
    return Error

#################################
# LeNet5 - Backward Propagation #
#################################

def LeNet5_backward(cache_C1, X_A1, cache_S2, cache_C3, X_A3, cache_S4, cache_C5, X_A5, X_A6, X_A7, X_A8, Y_pred, Y_truth, pool_mode = "avgpooling", act_mode = "sigmoid"):
    
    #D_A8 = Y_pred - Y_truth
    
    #D_Z8_local = activation_backward(X_A8, act_mode)
    
    #D_Z8 = np.multiply(D_Z8_local, D_A8)
    D_Z8 = Y_pred - Y_truth
    D_W8 = np.outer(X_A7, D_Z8)
    D_A7 = np.dot(D_Z8, D_W8.T)
    
    D_Z7_local = activation_backward(X_A7, act_mode)
    D_Z7 = np.multiply(D_Z7_local, D_A7)
    
    D_W7 = np.outer(X_A6, D_Z7)
    D_A6 = np.dot(D_Z7, D_W7.T)
    D_A5 = D_A6.reshape(1,1,120)
    
    D_C5_local = activation_backward(X_A5, act_mode)
    D_C5 = np.multiply(D_C5_local, D_A5)
    D_S4, D_K_C5, D_b_C5 = conv_backward(D_C5, cache_C5)
    D_A3 = pool_backward(D_S4, cache_S4, pool_mode)
    
    D_C3_local = activation_backward(X_A3, act_mode)
    D_C3 = np.multiply(D_C3_local, D_A3)
    D_S2, D_K_C3, D_b_C3 = conv_backward(D_C3, cache_C3)
    D_A1 = pool_backward(D_S2, cache_S2, pool_mode)
    
    D_C1_local = activation_backward(X_A1, act_mode)
    D_C1 = np.multiply(D_C1_local, D_A1)
    D_X, D_K_C1, D_b_C1 = conv_backward(D_C1, cache_C1)
    
    return D_W8, D_W7, D_K_C5, D_b_C5, D_K_C3, D_b_C3, D_K_C1, D_b_C1

def update_trainable_parameters(lr, D_W8, W8, D_W7, W7, D_K_C5, K_C5, D_b_C5, b_C5, D_K_C3, K_C3, D_b_C3, b_C3, D_K_C1, K_C1, D_b_C1, b_C1):
    
    W8 = W8 - lr * D_W8
    W7 = W7 - lr * D_W7
    K_C5 = K_C5 - lr * D_K_C5
    b_C5 = b_C5 - lr * D_b_C5
    K_C3 = K_C3 - lr * D_K_C3
    b_C3 = b_C3 - lr * D_b_C3
    K_C1 = K_C1 - lr * D_K_C1
    b_C1 = b_C1 - lr * D_b_C1
    
    return W8, W7, K_C5, b_C5, K_C3, b_C3, K_C1, b_C1


import time


# [Output Vars]
#   1. <int> top1_accuracy
#   2. <int> top5_accuracy
def top_accuracy(Metadata, Name, K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8, pool_mode, act_mode):
    
    num_top1_pred = 0
    num_top5_pred = 0
    len_dataset = len(Metadata)
    
    tic = time.time()
    for i in range(len_dataset):
        # 1. Read a specific image in RGB format.
        img = cv.imread(ROOT_PATH + Metadata[i][0])
        img_label = Metadata[i][1]
    
        # 2. Resize the image to a fixed size (128, 128)
        img_resize = cv.resize(img, (28, 28))
        X = img_resize / 255.0
        
        cache_C1, X_A1, cache_S2, cache_C3, X_A3, cache_S4, cache_C5, X_A5, X_A6, X_A7, X_A8, Y_pred = LeNet5_forward(X, K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8, pool_mode, act_mode)
        if i == 100  or i == 200: print(f"[{i}-th Prediction] Prediction:\n{Y_pred[0,:5]}")

        # 4. Grab top 5 predictions.
        top_1, top_2, top_3, top_4, top_5 = grab_top_5_predictions(Y_pred)
        
        # 5. Check if the label is the top 1 prediction.
        if img_label == top_1: num_top1_pred = num_top1_pred + 1

        # 6. Check if the label is in the top 5 predictions
        if img_label in [top_1, top_2, top_3, top_4, top_5]: num_top5_pred = num_top5_pred + 1
                 
    top1_accuracy = round(num_top1_pred/len_dataset*100, 2)
    top5_accuracy = round(num_top5_pred/len_dataset*100, 2)
    
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

ROOT_PATH = "C:/Users/USER/Desktop/Projects/Github_Repo/AI/DeepLearning/__HW1_DATA/"
NP_TRAIN_TXT, NP_TEST_TXT, NP_VAL_TXT = read_metadata_files(ROOT_PATH)

Metadata = NP_TRAIN_TXT

import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize Parameters
K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8 = Initialize_Parameters(-0.1, 0.1)

pool_mode = "maxpooling"
act_mode = "relu"
Epoch = 5
lr = 0.01

# 0. Shuffle the training dataset.
Length_TRAIN = len(Metadata)
Length_VAL = len(NP_VAL_TXT)
random_index = np.arange(Length_TRAIN)
np.random.shuffle(random_index)

val_top1_list = []
val_top5_list = []

tic = time.time()
val_top1, val_top5 = top_accuracy(NP_VAL_TXT, "Val", K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8, pool_mode, act_mode)
toc = time.time()
print(f"[Epoch: {-1}] The val top-1 Acc. is {val_top1}, val top-5 Acc. is {val_top5}.")
print(f"[Epoch: {-1}] [Measure Accuracy] Spend {round(toc-tic,2)} sec.\n")

val_top1_list.append(val_top1)
val_top5_list.append(val_top5)

loss_list = []
train_size = Length_TRAIN
for epoch in range(Epoch):
    tmp_list = []
    counter = 0
    forward_in_one_epoch = 0
    backward_in_one_epoch = 0
    for i in random_index[:train_size]:
    
        # 1. Read a specific image in RGB format.
        img = cv.imread(ROOT_PATH + Metadata[i][0])
        #print(img.shape)
        img_label = Metadata[i][1]
    
        # 2. Resize the image to a fixed size (28, 28)
        img_resize = cv.resize(img, (28, 28))
        X = img_resize / 255.0
        Y_truth = np.zeros((1,50))
        Y_truth[0][img_label] = 1
    
        # 3. Forward Pass
        tic = time.time()
        cache_C1, X_A1, cache_S2, cache_C3, X_A3, cache_S4, cache_C5, X_A5, X_A6, X_A7, X_A8, Y_pred = LeNet5_forward(X, K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8, pool_mode, act_mode)
        toc = time.time()
        forward_in_one_epoch = forward_in_one_epoch + (toc -tic)
        
        # 4. Cross Entropy Loss
        tmp_list.append(cross_entropy(Y_pred, Y_truth))
        
        # 5. Backward Pass
        tic = time.time()
        D_W8, D_W7, D_K_C5, D_b_C5, D_K_C3, D_b_C3, D_K_C1, D_b_C1 = LeNet5_backward(cache_C1, X_A1, cache_S2, cache_C3, X_A3, cache_S4, cache_C5, X_A5, X_A6, X_A7, X_A8, Y_pred, Y_truth, pool_mode, act_mode)
        toc = time.time()
        backward_in_one_epoch = backward_in_one_epoch + (toc-tic)
        
        # 6. Update Weights
        W8, W7, K_C5, b_C5, K_C3, b_C3, K_C1, b_C1 = update_trainable_parameters(lr, D_W8, W8, D_W7, W7, D_K_C5, K_C5, D_b_C5, b_C5, D_K_C3, K_C3, D_b_C3, b_C3, D_K_C1, K_C1, D_b_C1, b_C1)    
        
        if counter == 100 or counter == 200: 
            print(f"[Epoch: {epoch} || ({counter+1}/{train_size})] [Loss] {cross_entropy(Y_pred, Y_truth)} [Label]{img_label}\n[Check 1st D_K]:{D_K_C1[:,0,0,0]}")
        counter = counter + 1
    
    # 7. Measure Accuracy
    tic = time.time()
    val_top1, val_top5 = top_accuracy(NP_VAL_TXT, "Val", K_C1, b_C1, hparam_C1, hparam_S2, K_C3, b_C3, hparam_C3, hparam_S4, K_C5, b_C5, hparam_C5, W7, W8, pool_mode, act_mode)
    toc = time.time()

    print(f"[Epoch: {epoch}] [Forward Propagation] Spend {round(forward_in_one_epoch,2)} sec.")
    print(f"[Epoch: {epoch}] [Backward Propagation] Spend {round(backward_in_one_epoch,2)} sec.")
    print(f"[Epoch: {epoch}] The val top-1 Acc. is {val_top1}, val top-5 Acc. is {val_top5}.")
    print(f"[Epoch: {epoch}] The average loss is {np.mean(tmp_list)}]")
    print(f"[Epoch: {epoch}] [Measure Accuracy] Spend {round(toc-tic,2)} sec.")

    val_top1_list.append(val_top1)
    val_top5_list.append(val_top5)
    loss_list.append(np.mean(tmp_list))


plt.plot(val_top1_list)
plt.show()
plt.plot(val_top5_list)
plt.show()
plt.plot(loss_list)
plt.show()