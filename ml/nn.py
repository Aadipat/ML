import numpy as np
from random import random

#Neural Network Class
class NeuralNetwork():
    #Constructor
    def __init__(self, input_nodes = 2, hidden_layers = [2], output_nodes = 1):

        self.input_nodes = input_nodes
        self.hidden_layers = hidden_layers
        self.output_nodes = output_nodes

        layers = [input_nodes] + hidden_layers + [output_nodes]

        hidden_nodes = 0
        for i in hidden_layers:
            hidden_nodes += i

        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)
        self.weights = weights
        # print("Random starting weights: ");
        # print(self.weights);

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):

        activations = inputs
        self.activations[0] = activations

        for i, w in enumerate(self.weights):

            net_outputs = np.dot(activations, w)
            activations = self.sigmoid(net_outputs)
            self.activations[i + 1] = activations

        return activations

    def back_propagate(self,error):

        for i in reversed(range(len(self.derivatives))):

            activations = self.activations[i+1]
            delta = error * self.sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            self.derivatives[i] = np.dot(current_activations, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
            # print(self.derivatives[i])
        return error

    def gradient_descent(self, learningRate = 0.1):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learningRate

    def learn(self,error, learning_rate):
        self.back_propagate(error)
        self.gradient_descent(learning_rate)

    def train(self, inputs, targets, epochs, learning_rate = 0.1):

        for e in range(epochs):

            for i, input in enumerate(inputs):

                target = targets[i]
                output = self.think(input)
                error = target - output
                # print(error)
                self.learn(error, learning_rate)

#MAIN
nn = NeuralNetwork(2, [2], 1)

# Line and diagonal 2x2 matrix.
# inputs = np.array([[1,1,0,0],[1,0,1,0],[0,0,1,1],[0,1,0,1],[1,0,0,1],[0,1,1,0]])
# targets = np.array([[1,0],[1,0],[1,0],[1,0],[0,1],[0,1]])

inputs = np.array([[1,1],[1,0],[0,1],[0,0]])
targets = np.array([[0],[1],[1],[0]])
 
nn.train(inputs, targets, 100000, 0.1)

# TEST NETWORK
input = np.array([1,1])

output = nn.think(input)

print(output)

# print("I think that grid( {} {} {} {} ) is {} percent line  & {} percent diagonal".format(input[0], input[1], input[2], input[3], output[0], output[1]))
