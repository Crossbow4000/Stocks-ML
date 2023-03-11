import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy
import random
from data import GetTickerData
import shutup
from math import ceil
import plotly.express as px
import pandas as pd

shutup.please()

# dataframe = dict(
#     Epoch = [0],
#     Loss = [0]
# )
# figure = px.line(dataframe, x='Epoch', y='Loss', title='Loss Per Epoch')
# figure.show()

model = torch.nn.Sequential(
    torch.nn.Linear(120, 4000),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(4000, 2000),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(2000, 500),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(500, 1),
    torch.nn.Tanh()
)

model.load_state_dict(torch.load('./models/model.pt'))

learnRate = 0.0000001
epochs = 100
batchSize = 100

dtype = torch.float
device = torch.device("cpu")

print('Getting data . . .\n')

data = GetTickerData('AMZN')
dataLength = len(data[0])

lossFunction = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)

dataset = TensorDataset(torch.FloatTensor(data[0]), torch.FloatTensor(data[1]))
loader = DataLoader(dataset, shuffle=True, batch_size=batchSize)

data = None # Clear to save memory

lossesArray = []
epochsArray = []
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    averageLossArray = []
    for i, [input, expectedOutput] in enumerate(loader):
        # if len(input) != 120:
        #     continue

        input = torch.FloatTensor(input)
        expectedOutput = torch.FloatTensor(expectedOutput)

        prediction = model(input.unsqueeze(dim=0))
        loss = lossFunction(prediction, expectedOutput)
        averageLossArray.append(loss)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        print(f'\rTraining Sample : [{i}/{round(dataLength/batchSize)}]', end='')
    print(f'\rTraining Sample : [{i}/{round(dataLength/batchSize)}]', end='')

    averageLoss = (sum(averageLossArray))/(len(averageLossArray))
    lossesArray.append(averageLoss)
    epochsArray.append(epoch+1)

    # with torch.no_grad():
    #     dataframe = dict(
    #         Epoch = epochsArray,
    #         Loss = lossesArray
    #     )
    #     figure = px.line(dataframe, x='Epoch', y='Loss', title='Loss Per Epoch')
    #     figure.update()

    print(f'\nAverage Loss : {averageLoss}')
    print('-------------------------------\n')

    torch.save(model.state_dict(), './models/model.pt')
