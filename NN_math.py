"""
Created on Saturday Sept 16 14:37:20 2018

@author: Srishti
"""


import numpy as np
import pickle
import gzip
from sklearn import datasets
import matplotlib.pyplot as plt

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

# Helper function to predict an output (0 or 1)
def predict(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    hidden_layer_input1=np.dot(X,W1)
    hidden_layer_input=hidden_layer_input1 + b1
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,W2)
    output_layer_input= output_layer_input1+ b2
    output = sigmoid(output_layer_input)
    return np.argmax(output, axis=1)

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)


def main():
    #((X, y), (x_valid, y_valid), _) = pickle.load(gzip.open('C:\\Users\\kkrishna\\mnist.pkl.gz', 'rb'), encoding='latin-1')
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

        
    #x_valid = (x_valid-mean)/std

    #Variable initialization
    epoch=50000 #Setting training iterations
    lr=0.01 #Setting learning rate
    inputlayer_neurons = 2#X.shape[1] #number of features in data set
    hiddenlayer_neurons = 3 #number of hidden layers neurons
    output_neurons = 2 #number of neurons at output layer
    
    #weight and bias initialization
    wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
    bh=np.random.uniform(size=(1,hiddenlayer_neurons))
    wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
    bout=np.random.uniform(size=(1,output_neurons))

    #This is what we return at the end
    model = {}
    for i in range(epoch):
        #Forward Propogation
        hidden_layer_input1=np.dot(X,wh)
        hidden_layer_input=hidden_layer_input1 + bh
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,wout)
        output_layer_input= output_layer_input1+ bout
        output = sigmoid(output_layer_input)
        
        #Backpropagation
        E = y-output
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        wout += hiddenlayer_activations.T.dot(d_output) *lr
        bout += np.sum(d_output, axis=0,keepdims=True) *lr
        wh += X.T.dot(d_hiddenlayer) *lr
        bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

        # Assign new parameters to the model
        model = { 'W1': wh, 'b1': bh, 'W2': wout, 'b2': bout}
        if i % 1000 == 0:
            print("Loss after iteration %i: %f" %(i, (100 * np.mean(np.square(E)))))
        
    plt.scatter(X[:,0], X[:,1], c=y[:,0])
    # Plot the decision boundary
    plot_decision_boundary(lambda X: predict(model, X), X, y[:,0])
    plt.title("Decision Boundary for hidden layer size 3")
    

if __name__ == '__main__':
  main()
