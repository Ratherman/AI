import numpy as np
import matplotlib.pyplot as plt

# Function 【01】 ~ 【06】 are public functions.
    # 【01】forward_pass
    # 【02】backward_pass
    # 【03】cross_entropy
    # 【04】initialize_weights
    # 【05】update_weights
    # 【06】top_accuracy
    
# Function 【07】 ~ 【09】 are private functions.
    # 【07】_grab_top_5_predictions
    # 【08】_sigmoid
    # 【09】_softmax

class nn():

    # 【00】 #########################
    # [Input Vars] None
    #
    # [Output Vars] None
    def __init__(self):
        pass

    # 【01】 #########################
    # [Input Vars]
    #   1. <ndarray> X: Its shape should be (1, 769).
    #   2. <ndarray> W1: Its shape should be (769, 300).
    #   3. <ndarray> W2: Its shape should be (300, 50).
    #
    # [Output Vars] 
    #   1. <ndarray> S: Its shape should be (1, 50).
    #   2. <ndarray> a2: Its shape should be (1, 50).
    #   3. <ndarray> a1: Its shape should be (1, 300).
    def forward_pass(self, X, W1, W2):
        assert X.shape == (1, 769), f"[Error] X's shape is {X.shape}. Expected shape is (1, 769)."
        assert W1.shape == (769, 300), f"[Error] W1's shape is {W1.shape}. Expected shape is (769, 300)."
        assert W2.shape == (300, 50), f"[Error] W2's shape is {W2.shape}. Expected shape is (300, 50)."
        
        Z1 = np.dot(X, W1)
        assert Z1.shape == (1, 300), f"[Error] Z1's shape is {Z1.shape}. Expected shape is (1, 300)."
        
        A1 = self._sigmoid(Z1)
        assert A1.shape == (1, 300), f"[Error] A1's shape is {A1.shape}. Expected shape is (1, 300)."
        
        Z2 = np.dot(A1, W2)
        assert Z2.shape == (1, 50), f"[Error] Z2's shape is {Z2.shape}. Expected shape is (1, 50)."
        
        A2 = self._sigmoid(Z2)
        assert A2.shape == (1, 50), f"[Error] A2's shape is {A2.shape}. Expected shape is (1, 50)."

        Y_pred = self._softmax(A2)
        assert Y_pred.shape == (1, 50), f"[Error] Y_pred's shape is {S.shape}. Expected shape is (1, 50)."
        
        return Y_pred, A2, A1

    # 【02】 #########################
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
    def backward_pass(self, Y_pred, Y_truth, A2, A1, X, W2, W1):
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

    # 【03】 #########################
    # [Input Vars]
    #   1. <ndarray> Y_pred: Its shape should be (1, 50).
    #   2. <ndarray> Y_truth: Its shape should be (1, 50).
    #
    # [Output Vars]
    #   2. <ndarray> Error
    def cross_entropy(self, Y_pred, Y_truth):
        assert Y_truth.shape == (1, 50), f"[Error] Y_truth's shape is {Y_truth.shape}. Expected shape is (1, 50)."
        assert Y_pred.shape == (1, 50), f"[Error] Y_pred's shape is {S.shape}. Expected shape is (1, 50)."
        Error = (-1 * Y_truth * np.log(Y_pred)).sum()
        return Error

    # 【04】 #########################
    # [Input Vars] None
    #
    # [Output Vars]
    #   1. <ndarray> W1: Its shape should be (769, 300)
    #   2. <ndarray> W2: Its shape should be (300, 50)
    def initialize_weights(self):
        np.random.seed(0)
        W1 = np.random.uniform(low=-0.01, high=0.01, size=(769,300))
        W2 = np.random.uniform(low=-0.01, high=0.01, size=(300,50))
        assert W1.shape == (769, 300), f"[Error] W1's shape is {W1.shape}. Expected shape is (769, 300)."
        assert W2.shape == (300, 50), f"[Error] W2's shape is {W2.shape}. Expected shape is (300, 50)."
        return W1, W2

    # 【05】 #########################
    # [Input Vars]
    #   1. <ndarray> dEdW1
    #   2. <ndarray> dEdW2
    #   3. <ndarray> W1
    #   4. <ndarray> W2
    #   5. <float> lr
    #
    # [Output Vars]
    #   1. <ndarray>
    #   2. <ndarray>
    def update_weights(self, dEdW1, dEdW2, W1, W2, lr):
        W1 = W1 - lr * dEdW1
        W2 = W2 - lr * dEdW2
        return W1, W2

    # 【06】 #########################
    # [Input Vars]
    #   1. <list> Dataset: Either Train_COH_Dataset, Val_COH_Dataset, or Test_COH_Dataset
    #   2. <list> Label: Train_COH_Label, Val_COH_Label, Test_COH_Label
    #   3. <ndarry> W: It's the updated Weight from training process.
    #   4. <float> Scale: It's the hyper parameter we decided in training process.
    #   5. <String> Name: It's for convenient purpose
    #
    # [Output Vars]
    #   1. <int> top1_accuracy
    #   2. <int> top5_accuracy
    def top_accuracy(self, Dataset, Label, W1, W2, Scale, Name):

        num_top1_pred = 0
        num_top5_pred = 0
        len_dataset = len(Label)
        
        for i in range(len_dataset):
            # 1. Grab the i-th data
            X = np.array(Dataset[i:i+1]).reshape(1, 769)/Scale
            Y = int(Label[i:i+1][0])
            assert X.shape == (1,769), f"[Error] X's shape is {X.shape}. Expected shape is (1, 769)."
            
            # 2. Predict the label by using Softmax.
            Y_pred, A1, A2 = self.forward_pass(X, W1, W2)
            assert Y_pred.shape == (1,50) , f"[Error] Y_pred's shape is {Y_pred.shape}. Expected shape is (1, 50)."
            
            # 3. Grab top 5 predictions.
            top_1, top_2, top_3, top_4, top_5 = self._grab_top_5_predictions(Y_pred)

            # 4. Check if the label is the top 1 prediction.
            if Y == top_1: num_top1_pred = num_top1_pred + 1

            # 5. Check if the label is in the top 5 predictions
            if Y in [top_1, top_2, top_3, top_4, top_5]: num_top5_pred = num_top5_pred + 1
  
        top1_accuracy = round(num_top1_pred/len_dataset*100, 2)
        top5_accuracy = round(num_top5_pred/len_dataset*100, 2)
        print(f"[Result of {Name}] The top-1 accuracy is {top1_accuracy} %")
        print(f"[Result of {Name}] The top-5 accuracy is {top5_accuracy} %")
        return top1_accuracy, top5_accuracy

    # 【07】 #########################
    # [Input Vars]
    #   1. <ndarray> Y_pred: It's a 1-D ndarray which contains the possibilities of the predictions.
    #
    # [Output Vars]
    #   1. <int> top_1: The 1st likely breed among those 50 breeds.
    #   2. <int> top_2: The 2nd likely breed among those 50 breeds.
    #   3. <int> top_3: The 3rd likely breed among those 50 breeds.
    #   4. <int> top_4: The 4th likely breed among those 50 breeds.
    #   5. <int> top_5: The 5th likely breed among those 50 breeds.
    def _grab_top_5_predictions(self, Y_pred):
        
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

    # 【08】 #########################
    # [Input Vars]
    #   1. <ndarray> Z
    #
    # [Output Vars]
    #   1. <ndarray> A
    def _sigmoid(self, Z):
        A = 1/(1 + np.exp(-Z))
        return A

    # 【09】 #########################
    # [Input Vars]
    #   1. <ndarray> A
    #
    # [Output Vars]
    #   1. <ndarray> Y_pred
    def _softmax(self, A):
        Y_pred = np.exp(A-np.max(A))/np.sum(np.exp(A-np.max(A)))
        return Y_pred