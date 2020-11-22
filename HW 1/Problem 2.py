import torch
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1588390)


def shuffle(x, y):
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    return x[indices], y[indices]


# MARK: - Read data
healthy, ill = [], []
with open('data.csv', 'r') as f:
    for line in f.readlines():
        arr = list(map(float, line.split(',')))
        if arr[-1] == 1.:
            ill.append(arr[:-1])
        else:
            healthy.append(arr[:-1])

healthy = np.array(healthy, dtype=np.float32)
ill = np.array(ill, dtype=np.float32)


# MARK: - Experiment starts
for n in [40, 80, 120, 160, 200]:
    accuracy = []

    for _ in range(1000):
        np.random.shuffle(healthy)
        np.random.shuffle(ill)

        trainX = np.vstack((healthy[:n], ill[:n]))
        trainY = np.vstack((np.zeros((n, 1)), np.ones((n, 1)))).astype(np.float32)

        testX  = np.vstack((healthy[n:], ill[n:]))
        testY  = np.vstack((np.zeros((len(healthy) - n, 1)), np.ones((len(ill) - n, 1)))).astype(np.float32)

        trainX, trainY = shuffle(trainX, trainY)
        testX, testY = shuffle(testX, testY)

        x = torch.from_numpy(trainX)
        t = torch.from_numpy(trainY)
        xt = x.transpose(0, 1)

        w = torch.matmul(torch.matmul(torch.inverse(torch.matmul(xt, x)), xt), t)

        xTest = torch.from_numpy(testX)
        yTest = torch.from_numpy(testY)

        yPred = torch.matmul(xTest, w).numpy()
        yPred = np.where(yPred >= .5, 1, 0)

        accuracy.append(np.sum(yPred == testY) * 1. / len(testY))

    avgAcc = np.average(accuracy)

    plt.plot(n, avgAcc, 'o:g')

plt.xlabel('Sample size N')
plt.ylabel('Accuracy')
plt.savefig('result.png')
plt.show()
