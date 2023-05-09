from neuralNetwork import *
import settings
import time

# ToDo:
#  - try useful values/settings for hiddenNotes and the learning rate automatically
#  - save which values give the best results
#  - value photos from your own handwriting
#  - train and test the programm with the zalando database

settings = settings.Settings()

# read trainingsData
if settings.dataFile == "mnist_train_60000.csv":
    with open("data/mist_train_first_30000.csv") as f:
        trainingData = f.readlines()
    with open("data/mist_train_second_30000.csv") as f:
        trainingData += f.readlines()
else:
    with open(settings.pathDataFile) as f:
        trainingData = f.readlines()

# read testdata
with open(settings.pathTestDataFile) as f:
    testData = f.readlines()

startTime = time.time()

neuralNetwork = NeuralNetwork(settings.inputNodes, settings.hiddenNodes, settings.outNodes, settings.learingRate)

for epoche in range(settings.epochs):
    for imageData in trainingData:
        singleDataOfOneImage = imageData.split(",")
        # scale inputs because not grey 0 to 255 we need 0.01 to 1.0
        inputs = (np.asfarray(singleDataOfOneImage[1:]) / 255.0 * 0.99) + 0.01
        # create target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(settings.outNodes) + 0.01
        # all values[0] is the target label for this record
        targets[int(singleDataOfOneImage[0])] = 0.99
        neuralNetwork.train(inputs, targets)

learningSucess = []

for imageData in testData:
    singleDataOfOneImage = imageData.split(",")
    correctNumber = int(singleDataOfOneImage[0])  # correctNumber is the truth
    inputs = (np.asfarray(singleDataOfOneImage[1:]) / 255.0 * 0.99) + 0.01
    outputs = neuralNetwork.query(inputs)
    determinedNumberFromTheNeuralNetwork = np.argmax(outputs)
    if determinedNumberFromTheNeuralNetwork == correctNumber:
        learningSucess.append(1)
    else:
        learningSucess.append(0)
        pass
    pass

learningSucessArray = np.asarray(learningSucess)
endTime = time.time()
durationTime = endTime - startTime

print(f"performance = {(learningSucessArray.sum() / learningSucessArray.size) * 100.0} %")
print(f"with: hiddenNodes: {settings.hiddenNodes} learning rate: {settings.learingRate} epochs: {settings.epochs}")
print(f"Time taken: {durationTime:.03f} s   {durationTime / 60.0 :.03f} min")
