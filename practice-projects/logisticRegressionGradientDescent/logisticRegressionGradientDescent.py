import numpy as np
import h5py
import scipy
from scipy import ndimage
from PIL import Image

def sigmoid(x):
    return (1./(1. + np.exp(-x)))

# Binary Image Classification with Logistic Regression using Gradient Descent
class LogisticRegressionGD:
    def __init__(self, X_train, Y_train, X_test, Y_test, classes):
        """
        X_train: training set represented by a numpy array of shape (m_train,width,height,3[RGB])
        Y_train: training labels (classification) represented by a numpy array of shape (m_train,)
        X_test: test set represented by a numpy array of shape (m_test,width,height,3[RGB])
        Y_test: test labels represented by a numpy array of shape (m_test,)
        classes: list of classes (set of labels)
        """
        # mark width and height of training images before flattening set
        self.width = X_train.shape[1]
        self.height = X_train.shape[2]

        # training set
        # flatten to 1d array and standardize input
        X_train = (X_train.reshape(X_train.shape[0], -1).T) / 255
        X_test = (X_test.reshape(X_test.shape[0], -1).T) / 255
        self.X_train = X_train # numpy array of shape (width * height * 3, m_train)
        self.X_test = X_test # numpy array of shape (width * height * 3, m_test)

        # training labels
        # flatten
        Y_train = Y_train.reshape((1, Y_train.shape[0]))
        Y_test = Y_test.reshape((1, Y_test.shape[0]))
        self.Y_train = Y_train # numpy array of shape (1, m_train)
        self.Y_test = Y_test # numpy array of shape (1, m_test)

        # sanity check
        assert(X_train.shape[1] == Y_train.shape[1])
        assert(X_test.shape[1] == Y_test.shape[1])

        # m training examples
        self.m_train = X_train.shape[1]
        # m test examples
        self.m_test = X_test.shape[1]

        # classes
        self.classes = classes

        # initialize with zeros
        # weights: a numpy array of size (width * height * 3, 1)
        self.weights = np.zeros((X_train.shape[0],1))
        # bias: a scalar
        self.bias = 0.

        # keep track of cost of each iteration
        self.costs = []

    def train(self, num_iterations = 2000, learning_rate = 0.005):
        """
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        """
        # train: learn (optimize) weights and bias
        for i in range(num_iterations):
            # Cost and gradient calculation
            # forward propagation (from x to cost)
            # Z = w.T*X + b
            Z = np.dot(self.weights.T, self.X_train) + self.bias
            # A: activation, probabilities of a label being present,
            # i.e. output of logistic regression
            A = sigmoid(Z)

            # backward propagation (to find gradient)
            dZ = A - self.Y_train
            # dw: gradient of the loss with respect to weights, thus same shape as weights
            dw = (1/self.m_train) * np.dot(self.X_train, dZ.T)
            # db: gradient of the loss with respect to bias, thus same shape as bias
            db = (1/self.m_train) * np.sum(dZ.T)

            # update weights and bias with gradient descent
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            # compute cost: negative log-likelihood cost for logistic regression
            Loss = -1 * (self.Y_train * np.log(A) + (1 - self.Y_train) * np.log(1 - A))
            cost = (1/self.m_train) * np.sum(Loss)
            self.costs.append(cost)

        # measure prediction on training set
        A = sigmoid(np.dot(self.weights.T, self.X_train) + self.bias)
        Y_prediction_train = np.where(A > 0.5, 1, 0)

        # measure prediction on test set
        A = sigmoid(np.dot(self.weights.T, self.X_test) + self.bias)
        Y_prediction_test = np.where(A > 0.5, 1, 0)

        training_accuracy = 100 - np.mean(np.abs(Y_prediction_train - self.Y_train)) * 100
        test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - self.Y_test)) * 100

        return { "costs": self.costs,
                 "Y_prediction_test": Y_prediction_test,
                 "Y_prediction_train" : Y_prediction_train,
                 "weights" : self.weights,
                 "bias" : self.bias,
                 "training_accuracy": training_accuracy,
                 "test_accuracy": test_accuracy }

    def predict(self, imageFile):
        # read image into np array
        image = np.array(ndimage.imread(imageFile, flatten=False))
        # resize image to match training examples, flatten
        reshape_image = scipy.misc.imresize(image, size=(self.width,self.height)).reshape((1, self.weights.shape[0])).T
        # compute activation
        A = sigmoid(np.dot(self.weights.T, reshape_image) + self.bias)
        # compute prediction
        Y = np.where(A > 0.5, 1, 0)
        classification = np.squeeze(Y)
        print("Predicted classification: " + classes[np.squeeze(Y)].decode("utf-8"))

if __name__ == "__main__":
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    # train set features
    train_set_x = np.array(train_dataset["train_set_x"][:])
    # train set labels
    train_set_y = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
    # test set features
    test_set_x = np.array(test_dataset["test_set_x"][:])
    # test set labels
    test_set_y = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    model = LogisticRegressionGD(train_set_x, train_set_y, test_set_x, test_set_y, classes)
    results = model.train()

    print("Training accuracy: %f, test accuracy: %f" %(results["training_accuracy"], results["test_accuracy"]))

    model.predict("datasets/test_image.jpg")
