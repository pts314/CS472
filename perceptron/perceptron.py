import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
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
        rows,cols = X.shape
        self.X = np.append(X, np.ones([rows,1]), axis=1)
        self.y = y
        self.c = .2
        w_rows,w_cols = initial_weights.shape
        self.weights = self.initialize_weights() if not initial_weights else initial_weights
        if (w_cols < cols+1) {
            print("Improperly sized weights. Using defaults")
            self.weights = self.initialize_weights()
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        rows = X.shape[0]
        y = [[]];
        for row in X:
            

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        rows,cols = self.X.shape
        return np.zeros(cols)

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        things = np.append(X, y, axis=1)
        np.random.shuffle(things)
        np.X = things[:,:-1]
        np.y = things[:,-1:]

    """
        runs through one epoch and updates the weights accordingly

    """
    def _train_one_epoch(self):
        changed=0
        for row, true_val in zip(self.X, self.y):
            predict,net = _predict_one_row(self, row)
            if (predict!=true_val):
                changed=1
                _update_weights(self,true_val-predict, row)
        return changed


    def _predict_one_row(self, row):
        threshold = 0;
        net = sum(row*self.weights)
        if (net > threshold): 
            return 1
        else
            return 0

    def _update_weights(self, changeDirection, row):
        delta = self.c*changeDirection
        self.weights+=row*delta

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights
