from numpy import *

def loadDataSet() :
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines() :
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX) :
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels) :
    # convert to NumPy matrix
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 3
    weights = ones((n, 1))
    # heavy on matrix operations
    for k in range(maxCycles) :
        mult = dataMatrix * weights
        print "\nmult", mult
        # matrix mult
        h = sigmoid(mult)
        # matrix subtraction
        error = (labelMat - h)
        # matrix mult
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def gradAscent2(dataMatIn, classLabels) :
    maxCycles = 20
    rMult = []
    rH = []
    rError = []
    rWeights = []    
    # convert to NumPy matrix
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    weights = ones((n, 1))
    # heavy on matrix operations
    for k in range(maxCycles) :
        mult = dataMatrix * weights
        # matrix mult
        h = sigmoid(mult)
        # matrix subtraction
        error = (labelMat - h)
        # matrix mult
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights) :
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n) :
        if int(labelMat[i]) == 1 :
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else :
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def test() :
    dataMat,labelMat = loadDataSet()
    weights = gradAscent2(dataMat,labelMat)
    plotBestFit(weights.getA())

test()

