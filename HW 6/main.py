from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from glob import glob
from PIL import Image
import numpy as np

np.random.seed(1588390)


def loadData():
    base = './att_faces_10/'
    train, val = [], []
    for i in range(1, 11):
        path = base + 's{}/*.pgm'.format(i)
        files = np.array(glob(path))
        np.random.shuffle(files)

        trainData, valData = [], []
        for f in files[:8]:
            img = np.array(Image.open(f)).flatten()
            trainData.append(img)

        for f in files[8:]:
            img = np.array(Image.open(f)).flatten()
            valData.append(img)

        train.append(trainData)
        val.append(valData)
    return np.array(train), np.array(val)


plt.figure(dpi=300)
ds = [1, 2, 3, 6, 10, 20, 30]
pcaAvgAcc, fldAvgAcc = [], []

for d in ds:
    pcaAcc, fldAcc = [], []

    for i in range(20):
        t, v = loadData()

        classes, _, features = t.shape
        y = [idx for idx in range(classes) for _ in range(8)]

        # MARK: train
        pca = PCA(n_components=d)
        pca40 = PCA(n_components=40)
        fld = LinearDiscriminantAnalysis(n_components=d)

        pcaX = pca.fit_transform(t.reshape((-1, features)))
        fldX = fld.fit_transform(pca40.fit_transform(t.reshape((-1, features))), y)

        # MARK: val
        pcaKnn, fldKnn = KNeighborsClassifier(1), KNeighborsClassifier(1)
        pcaKnn.fit(pcaX, y)
        fldKnn.fit(fldX, y)
        pcaCorrect, fldCorrect, length = 0, 0, 0

        for j in range(len(v)):
            pcaPred = pca.transform(v[j])
            pca40Pred = pca40.transform(v[j])
            fldPred = fld.transform(pca40Pred)

            pcaRes = pcaKnn.predict(pcaPred)
            fldRes = fldKnn.predict(fldPred)

            pcaCorrect += np.sum(pcaRes == j)
            fldCorrect += np.sum(fldRes == j)
            length += len(pcaRes)

        pcaAcc.append(pcaCorrect * 1. / length)
        fldAcc.append(fldCorrect * 1. / length)

    pcaAvgAcc.append(np.mean(pcaAcc))
    fldAvgAcc.append(np.mean(fldAcc))


plt.plot(ds, pcaAvgAcc, 'ro-', label='PCA')
plt.plot(ds, fldAvgAcc, 'gx-', label='FLD')
plt.xlabel('n_components')
plt.ylabel('accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('result.png')
