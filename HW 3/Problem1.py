from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from utils import plotConfusionMatrix
import numpy as np
import keras

np.random.seed(1588390)


# MARK: - load data
(xTrain, yTrain), (xTest, yTest) = keras.datasets.mnist.load_data()

xTrain = np.reshape(xTrain, (-1, 28*28)) / 255.
xTest = np.reshape(xTest, (-1, 28*28)) / 255.


# MARK: - logistic regression
model = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=100, verbose=2)
model.fit(xTrain, yTrain)

yPred = model.predict(xTest)
confusionMatrix = confusion_matrix(yTest, yPred)
plotConfusionMatrix(confusionMatrix, [i for i in range(10)], 'problem1.png')
