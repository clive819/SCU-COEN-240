import matplotlib.pyplot as plt
import numpy as np
import math
import sys


np.random.seed(1588390)
EPSILON = 1e-5


# MARK: - read data
numClusters = 3
data = []
clusters = [[] for _ in range(numClusters)]
with open('data.csv', 'r') as f:
    for line in f.readlines():
        l = list(map(float, line.split(',')))
        data.append(l[:-1])
        clusters[int(l[-1])].append(l[:-1])

data = np.array(data)
clusters = np.array(clusters)


# MARK: - init center
selected = np.random.choice(len(data), numClusters)
centroids = data[selected]


# MARK: - function define
def distance(a1, a2):
    ans = 0.
    for d in (a1 - a2):
        ans += d ** 2
    return math.sqrt(ans)


def findClosestCenter(arr):
    idx, minDistance = 0, sys.maxint
    for i in range(numClusters):
        d = distance(arr, centroids[i])
        if d < minDistance:
            idx, minDistance = i, d
    return idx


def J(center, cluster):
    ans = 0.
    for i in range(numClusters):
        for j in range(len(cluster[i])):
            for d in (cluster[i][j] - center[i]):
                ans += d ** 2
    return ans


# MARK: - K-Means
iteration = 0
prevJ = J(centroids, clusters)
plt.plot(iteration, prevJ, '*-g')
while True:
    # MARK: - (Step 1) assign to clusters
    newClusters = [[] for _ in range(numClusters)]
    for vector in data:
        c = findClosestCenter(vector)
        newClusters[c].append(vector)

    newClusters = np.array(newClusters)

    # MARK: - (Step 2) update centroids
    for i in range(numClusters):
        z = zip(*newClusters[i])
        for j in range(len(centroids[i])):
            centroids[i][j] = np.average(z[j])

    # MARK: - check if meets stopping criterion
    iteration += 1
    clusters = newClusters
    newJ = J(centroids, clusters)
    plt.plot(iteration, newJ, '*-g')
    if prevJ - newJ < EPSILON:
        break
    prevJ = newJ

plt.xlabel('Iteration')
plt.ylabel('J')
plt.savefig('result.png')
plt.show()
