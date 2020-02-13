import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

class MLPClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True,deterministic=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hid_layer_widths = np.array(hidden_layer_widths)
        self.n_hid_layers = len(hidden_layer_widths)
        self.n_layers = self.n_hid_layers+1
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.n_inputs = X.shape[1]
        self.n_outputs = y.shape[1]
        self.weights = self.initialize_weights() if not initial_weights else initial_weights
        self.lastDelta = []

        dont_stop = True
        iter_number = 0
        best_iter = 0
        scores = []
        best_weights = self.get_weights()
        max_flat_iter = 10
        while dont_stop:
            changed = self._train_one_epoch(X,y)
            scores.append(self.score(X,y))
            if scores[iter_number] > scores[best_iter]:
                best_iter = iter_number
                best_weights = self.get_weights().copy() # if you don't copy, it puts a reference to the object in, which will be changed
            # optionally add an epsilon for if very small difference between best and this
            changed_recently = (iter_number - best_iter) < max_flat_iter
            
            dont_stop = changed and changed_recently
            if self.shuffle:
                self._shuffle_data(X,y)
            iter_number +=1

        self.weights = best_weights
        if not quiet:
            print(f"ran {iter_number} iterations")

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        for row in X:
            y.append(self._predict_one(row))
        y = np.array(y)[0]
        return y[:, np.newaxis]

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        #self.hid_layer_widths = np.array(hidden_layer_widths)
        #self.n_hid_layers = len(hidden_layer_widths)
        #self.n_layers = self.n_hid_layers+1
        #self.n_inputs = X.shape[1]
        #self.n_outputs = y.shape[1]
        weights = []
        print(self.n_inputs)
        print(self.hid_layer_widths)
        print(self.n_outputs)
        layers = np.append(np.append(self.n_inputs, self.hid_layer_widths), [self.n_outputs])
        for i in range(self.n_layers):
            weights.append(np.ones([layers[i]+1, layers[i+1]]))
        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """  
        predicts = self.predict(weightedX)
        correct = 0;
        tot = 0;
        for predict,trueVal in zip(predicts,y):
            tot +=1
            if (predict==trueVal):
                correct += 1
        return correct*1.0/tot

#instead of thinking per node, it might be better to think of it per layer
# to get each output, multiply by the outputs of the last layer by the weights of the current layer
# O[N_l] * W[N_l][N_c] = N[N_c] (size of final output)
# N[N_c -> activation function -> O[N_c]
# back propegation:
# C*O*delta
# delta = sum of (delta_k * weight_j_k) * net(j)_prime
    def _train_one_epoch(self, X, y):
        changed=0
        for row, true_val in zip(X, y):
#TODO make this mlp not perceptron
            predict = self._predict_one(row)
            if (predict!=true_val):
                changed=1
                delta=true_val-predict
                self._update_weights(delta, row)
        return changed

        

    def _predict_one(self, inputs):
        # self.weights
        nets = [None]*self.n_layers
        intermediates = [None]*self.n_layers #np.array([nLayers+1]) 
        intermediates[0] = np.array(inputs)
        print(f"inputs are {inputs}")
        # if one layer:
        for i in range(self.n_layers):
            intermediates[i] = np.append(intermediates[i],1) # add bias
            print(i)
            print(intermediates[i])
            print(self.weights[i])
            nets[i] = np.matmul(intermediates[i],self.weights[i])
            intermediates[i+1] = self._activation(nets[i])
        output = intermediates[-1]
        return output, intermediates, nets
    
# i : previous stage, j : current node, k : next node
    def _back_prop_one(self, intermediates, nets):
        delta_w = [None]*self.n_layers
        #update output layer
        delta_j = (target-intermediates[-1])*self._prime(net[-1])
        delta_w[-1] = self.lr*intermediates[-2]*delta_j + self.momentum*self.lastDelta[-1]

        #update hidden layers
        for i in range(self.hiddenLayers-1, -1, -1):
            delta_k = delta_w[i+1]
            delta_j = sum(delta_k*self.weights[i+1]) * self._prime(net[i])
            delta_w[i] = self.lr*intermediates[i-1]*delta_j + self.momentum*self.lastDelta[i]

        return delta_w
       
       
    _activation = np.vectorize(lambda x: 1.0/(1+np.exp(-x)))
    _prime = np.vectorize(lambda x: x(1-x))

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        things = np.append(X, y, axis=1)
        thing = np.random.shuffle(things)
        np.X = things[:,:-1]
        np.y = things[:,-1:]


    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
