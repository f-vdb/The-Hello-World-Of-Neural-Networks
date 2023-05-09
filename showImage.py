import matplotlib.pyplot as plt
import numpy as np

with open("data/mnist_test_10.csv") as f:
    dataImages = f.readlines()


def plotImage(indexOfImage):
    if indexOfImage < 0 or indexOfImage > 99:
        print("wrong index....")
        return

    dataImageFromIndex = dataImages[indexOfImage].split(",")
    imageData = dataImageFromIndex[1:]  # drop the frist element (number), because it is the number
    imageArray = np.asfarray(imageData).reshape((28, 28))
    plt.imshow(imageArray, cmap="Greys", interpolation="None")
    plt.show()

    # we need the grey values not from 0 to 255 instead from 0.01 to 1.0
    # 0.01 because we don't want an input as zero

    scaledInput = (np.asfarray(dataImageFromIndex[1:]) / 255.0 * 0.99) + 0.01
    print(scaledInput)

for indexOfImage, _ in enumerate(dataImages):
    plotImage(indexOfImage)