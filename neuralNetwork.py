
import numpy as np
import scipy.special # sigmoid-funktion

class NeuralNetwork:
    # initialise the neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # set the number of nodes in each input, hidden and output layer
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate
        # w Matrix der Gewichte
        # i Matrix der Eingabewerte
        # w i h input hidden
        # w h o hidden output
        # easy
        # self.wih = (numpy.random.rand(self.hiddenNodes, self.inputNodes) - 0.5)
        # self.who = (numpy.random.rand(self.outputNodes, self.hiddenNodes) - 0.5)
        # with Stichproben der Gewichte aus einer Normalverteilung mit dem Mittelwert null und einer Standardabweichung
        #, die sich die Anzahl der Verknüpfungen zu einem Konten bezieht: 1/sqrt(Anzahl der eingehenden Verknüpfungen)
        self.wih = np.random.normal(0.0, pow(self.hiddenNodes, -0.5), (self.hiddenNodes, self.inputNodes))
        self.who = np.random.normal(0.0, pow(self.outputNodes, -0.5), (self.outputNodes, self.hiddenNodes))
        # The expit function, also known as the logistic sigmoid function is defined as expit(x) = 1/(1+exp(-x)).
        self.activationFunction = lambda x: scipy.special.expit(x)


    # train the neural network
    def train(self, inputsList, targetsList):
        # convert inputs list to 2d array
        inputs = np.array(inputsList, ndmin=2).T
        targets = np.array(targetsList, ndmin=2).T

        # calculate signals into hidden layer
        hiddenInputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hiddenOutputs = self.activationFunction(hiddenInputs)
        #calculate signals into final output layer
        finalInputs = np.dot(self.who, hiddenOutputs)
        # calculate the signals emerging from final output layer
        finalOutputs = self.activationFunction(finalInputs)
        # output layer error is the target - actual
        outputErrors = targets - finalOutputs
        # hidden layer error is the outputErrors, split by the weights, recombined at hiddenNodes
        hiddenErrors = np.dot(self.who.T, outputErrors)
        # update the weights for the links between the hidden and output layers
        # numpy.transpose returns an array with axes transposed.
        self.who += self.learningRate * np.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), np.transpose(hiddenOutputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.learningRate * np.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), np.transpose(inputs))

    # query the neural network
    def query(self, inputsList):
        # convert inputs list to 2d array
        inputs = np.array(inputsList, ndmin=2).T
        #calculate signals into hidden layer
        hiddenInputs = np.dot(self.wih, inputs)
        hiddenOutputs = self.activationFunction(hiddenInputs)
        # calculate signals into final output layer
        finalInputs = np.dot(self.who, hiddenOutputs)
        # calculate the signals emerging form final output layer
        finalOutputs = self.activationFunction(finalInputs)
        return finalOutputs