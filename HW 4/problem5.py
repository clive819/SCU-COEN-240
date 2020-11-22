from matplotlib import pyplot as plt
import numpy as np
import cv2


def calculateChromaticity(img):
    sumBGR = np.sum(img, axis=-1, dtype=np.float32) + 1e-10
    return img[:, :, 2] / sumBGR, img[:, :, 1] / sumBGR


# MARK: - read data
xTrain = cv2.imread('family.jpg')
yTrain = cv2.imread('family.png', cv2.IMREAD_GRAYSCALE)

xVal = cv2.imread('portrait.jpg')
yVal = cv2.imread('portrait.png', cv2.IMREAD_GRAYSCALE)


threshold = np.sum(yTrain == 0) / np.sum(yTrain == 255)

r, g = calculateChromaticity(xTrain)

# MARK: - calculate Gaussian Coefficients
r0, r1 = r[yTrain == 0], r[yTrain == 255]
g0, g1 = g[yTrain == 0], g[yTrain == 255]

u0r, u1r = np.mean(r0), np.mean(r1)
u0g, u1g = np.mean(g0), np.mean(g1)

v0r, v1r = np.mean((r0 - u0r) ** 2), np.mean((r1 - u1r) ** 2)
v0g, v1g = np.mean((g0 - u0g) ** 2), np.mean((g1 - u1g) ** 2)


# MARK: - test
R, G = calculateChromaticity(xVal)

P0r = (1. / np.sqrt(v0r * 2. * np.pi)) * np.exp(-1 / 2. * ((R - u0r) ** 2) / v0r)
P1r = (1. / np.sqrt(v1r * 2. * np.pi)) * np.exp(-1 / 2. * ((R - u1r) ** 2) / v1r)

P0g = (1. / np.sqrt(v0g * 2. * np.pi)) * np.exp(-1 / 2. * ((G - u0g) ** 2) / v0g)
P1g = (1. / np.sqrt(v1g * 2. * np.pi)) * np.exp(-1 / 2. * ((G - u1g) ** 2) / v1g)

P0 = P0r * P0g
P1 = P1r * P1g

result = np.where(P1 / P0 < threshold, 0, 255).astype(np.uint8)


# MARK: - benchmark
TPR = np.sum(yVal[result == yVal] == 255, dtype=np.float32) / np.sum(yVal == 255, dtype=np.float32) * 100.
TNR = np.sum(yVal[result == yVal] == 0, dtype=np.float32) / np.sum(yVal == 0, dtype=np.float32) * 100.

FPR = np.sum(yVal[result != yVal] == 0, dtype=np.float32) / np.sum(yVal == 0, dtype=np.float32) * 100.
FNR = np.sum(yVal[result != yVal] == 255, dtype=np.float32) / np.sum(yVal == 255, dtype=np.float32) * 100.


# MARK: - show result
f = plt.figure()
f.suptitle('TPR: {:0.2f}%, TNR: {:0.2f}%, FPR: {:0.2f}%, FNR: {:0.2f}%'.format(TPR, TNR, FPR, FNR))

f.add_subplot(1, 3, 1)
plt.imshow(xVal[:, :, ::-1])
plt.axis('off')
plt.title('Test Image')

f.add_subplot(1, 3, 2)
plt.imshow(yVal, cmap='gray')
plt.axis('off')
plt.title('Ground Truth Mask')

f.add_subplot(1, 3, 3)
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.title('Classification Result')

plt.tight_layout()
plt.savefig('result.png')
