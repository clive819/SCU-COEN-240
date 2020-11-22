from sklearn.metrics import confusion_matrix
from utils import plotConfusionMatrix
import torch.nn as nn
import numpy as np
import torchvision
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1588390)


# MARK: - load data
trainDataset = torchvision.datasets.MNIST('MNIST',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

testDataset = torchvision.datasets.MNIST('MNIST',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
testLoader = torch.utils.data.DataLoader(testDataset, batch_size=32, shuffle=True)


# MARK: - define neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.module = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.LogSoftmax(-1)
        )

    def forward(self, x):
        (b, _, _, _) = x.shape
        return self.module(x.view(b, -1))


model = Network().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# MARK: - train
model.train()
for epoch in range(5):
    for batch, (xTrain, yTrain) in enumerate(trainLoader):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        optimizer.zero_grad()
        out = model(xTrain)
        loss = nn.functional.nll_loss(out, yTrain)
        loss.backward()
        optimizer.step()

        print('Epoch: {}, Batch: {}, Loss: {:.2f}'.format(epoch, batch, loss.item()))

torch.save(model, 'Problem2.pt')


# MARK: - test
model.eval()
predictions, groundtruths = [], []

with torch.no_grad():
    for xTest, yTest in testLoader:
        out = model(xTest.to(device))
        yPred = out.argmax(-1).cpu().numpy().astype(np.uint8)

        predictions = np.hstack((predictions, yPred))
        groundtruths = np.hstack((groundtruths, yTest.numpy().astype(np.uint8)))

confusionMatrix = confusion_matrix(groundtruths, predictions)
plotConfusionMatrix(confusionMatrix, [i for i in range(10)], 'problem2.png')
