"""
Created on Saturday Sept 16 14:37:20 2018

@author: Srishti
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def sigmoid( s):
     # activation function
     return 1/(1+np.exp(-s))

# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, bh, W2, bout = model['W1'], model['bh'], model['W2'], model['bout']
    z = np.dot(X, W1) + bh # dot product of X (input) and first set of 3x2 weights
    z2 = sigmoid(z) # activation function
    z3 = np.dot(z2, W2) + bout # dot product of hidden layer (z2) and second set of 3x1 weights
    output = sigmoid(z3) # final activation function
    return np.argmax(output, axis=1)

# Helper function to plot decision boundary for the prediction dataset
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

class Neural_Network(object):
    def __init__(self):        
        #parameters
        self.inputSize = 2
        self.outputSize = 2
        self.hiddenSize = 3
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
        self.bh=np.random.uniform(size=(1,self.hiddenSize)) # (1X3) hidden layer bias
        self.bout=np.random.uniform(size=(1,self.outputSize))
        self.lr=.01
        self.model = { 'W1': self.W1, 'bh': self.bh, 'W2': self.W2, 'bout': self.bout}
        
    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) + self.bh# dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) + self.bout# dot product of hidden layer (z2) and second set of 3x1 weights
        output = self.sigmoid(self.z3) # final activation function
        return output
    
    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))
    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, output):
        # backward propagate through the network
        self.o_error = y - output # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(output) # applying derivative of sigmoid to error
        
        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) *self.lr # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) *self.lr # adjusting second set (hidden --> output) weights
        self.bout += np.sum(self.o_delta, axis=0,keepdims=True) *self.lr
        self.bh += np.sum(self.z2_delta, axis=0,keepdims=True) *self.lr

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self, xPredicted):
        return self.forward(xPredicted)
#        print("Predicted data based on trained weights: ");
#        print("Input (scaled): \n" + str(xPredicted));
#        print("Output: \n" + str(self.forward(xPredicted)));

    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
        model = { 'W1': self.W1, 'bh': self.bh, 'W2': self.W2, 'bout': self.bout}

def main():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    mean = X.mean()
    std = X.std()
    X=(X-mean)/std
    
    y = np.c_[ y, np.ones(200) ]
    rows = y.shape[0]
    for j in range(0, rows - 1):
        if y[j,0] == 0:
            y[j,1] = 1
        elif y[j,0] == 1:
            y[j,1] = 0 
    
    NN = Neural_Network()
    
    for i in range(50000): # trains the NN 1,000 times
        #print("Input: \n" + str(X))
        #print("Actual Output: \n" + str(y[:,0]))
        #print("Predicted Output: \n" + str(NN.forward(X)))
        #print("Loss: \n" + str(np.mean(np.square(y[:,0] - NN.forward(X))))) # mean sum squared loss
        NN.train(X, y)
        if i % 1000 == 0:
            print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
    
    NN.saveWeights()
    
    plt.scatter(X[:,0], X[:,1], c=y[:,0])
    # Plot the decision boundary
    plot_decision_boundary(lambda X: predict(NN.model, X), X, y[:,1])
    plt.title("Decision Boundary for hidden layer size 3")
    
if __name__ == '__main__':
  main()
