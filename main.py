import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy
import random
from data import GetTickerData
import shutup

shutup.please()

# Define the neural network architecture
model = torch.nn.Sequential(
    torch.nn.Linear(120, 3000),
    torch.nn.Sigmoid(),
    torch.nn.Linear(3000, 1),
    torch.nn.Tanh()
)

# Load the pre-trained model
model.load_state_dict(torch.load('./models/model.pt'))

# Set the hyperparameters for training
learnRate = 0.0000005
epochs = 100
batchSize = 277

dtype = torch.float
device = torch.device("cpu")

print('Getting data\n')

# Load the data
data = GetTickerData('NVDA')

# Define the loss function and optimizer
lossFunction = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)

# Train the model
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    averageLoss = []
    for [i, input], expectedOutput in zip(enumerate(data[0]), data[1]):
        # if len(input) != 120:
        #     continue

        input = torch.FloatTensor(input)
        expectedOutput = torch.FloatTensor(expectedOutput)

        prediction = model(input.unsqueeze(dim=0))
        loss = lossFunction(prediction, expectedOutput)
        averageLoss.append(loss)
        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if i % batchSize == 0:
            print(f'\rTraining Sample : [{i}/{len(data[0])}]', end='')
    print(f'\rTraining Sample : [{len(data[0])}/{len(data[0])}]', end='')

    print(f'\nAverage Loss : {(sum(averageLoss))/(len(averageLoss))}')
    print('-------------------------------\n')

    # Save the trained model
    torch.save(model.state_dict(), './models/model.pt')
