

class Settings:
    def __init__(self):
        self.inputNodes = 784   # 28 pixel x 28 pixel = 784
        self.hiddenNodes = 200 # try something but greater than 784 macht keinen sinn
        self.outNodes = 10      # the values 0,1,2,3,4,5,6,7,8,9
        self.learingRate = 0.3
        self.epochs = 3
        #self.testDataFile = "mnist_test_10.csv"
        #self.dataFile = "mnist_train_100.csv"
        self.testDataFile = "mnist_test_10000.csv"
        self.dataFile = "mnist_train_60000.csv" # datafile with 60000 numbers was to big for github > 100 mb
        self.directory = "data/"
        self.pathTestDataFile = self.directory + self.testDataFile
        self.pathDataFile = self.directory + self.dataFile



