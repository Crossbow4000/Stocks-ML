if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import numpy
    import random
    from data import GetTickerData, GetMostRecentTickerData
    import shutup
    from math import floor
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os

    shutup.please()

    task = 'train'

    model = torch.nn.Sequential(
        torch.nn.Linear(120, 4000),
        torch.nn.Softplus(),
        torch.nn.Linear(4000, 2000),
        torch.nn.Softplus(),
        torch.nn.Linear(2000, 500),
        torch.nn.Softplus(),
        torch.nn.Linear(500, 2),
        torch.nn.Sigmoid()
    )

    model.load_state_dict(torch.load('./models/model.pt'))

    dtype = torch.float
    device = torch.device("cpu")

    if task == 'train' or task == 'test':
        print('Getting data . . .\n')
        data = [[], []]
        newData = GetTickerData('EURUSD=X'); data[0].extend(newData[0]); data[1].extend(newData[1])
        newData = GetTickerData('GBPUSD=X'); data[0].extend(newData[0]); data[1].extend(newData[1])
        newData = GetTickerData('USDJPY=X'); data[0].extend(newData[0]); data[1].extend(newData[1])
        newData = GetTickerData('CHFUSD=X'); data[0].extend(newData[0]); data[1].extend(newData[1])
        newData = GetTickerData('GBPJPY=X'); data[0].extend(newData[0]); data[1].extend(newData[1])
        newData = GetTickerData('NZDUSD=X'); data[0].extend(newData[0]); data[1].extend(newData[1])
        newData = GetTickerData('CADJPY=X'); data[0].extend(newData[0]); data[1].extend(newData[1])
        newData = GetTickerData('AUDUSD=X'); data[0].extend(newData[0]); data[1].extend(newData[1])

        dataLength = len(data[0])
        print('')

        learnRate = 0.0000001
        epochs = 60
        batchSize = 1

        dataset = TensorDataset(torch.FloatTensor(data[0]), torch.FloatTensor(data[1]))
        loader = DataLoader(dataset, shuffle=True, batch_size=batchSize)

        lossFunction = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)

    data = None # Clear to save memory
    dataset = None # Clear to save memory

    # Train [Comment out to stop training]
    if task == 'train':
        plt.style.use('ggplot'); plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.ion()
        lossesArray = []
        epochsArray = []
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            averageLossArray = []
            for i, [input, expectedOutput] in enumerate(loader):
                input = torch.FloatTensor(input)
                expectedOutput = torch.FloatTensor(expectedOutput)

                prediction = model(input)
                loss = lossFunction(prediction, expectedOutput)
                averageLossArray.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                print(f'\rTraining Sample : [{i}/{floor(dataLength/batchSize)}]', end='')
            print(f'\rTraining Sample : [{i}/{floor(dataLength/batchSize)}]', end='')

            averageLoss = (sum(averageLossArray))/(len(averageLossArray))
            lossesArray.append(averageLoss)
            epochsArray.append(epoch+1)

            with torch.no_grad():
                movingAverage = []
                for i in range(0, len(lossesArray)):
                    length = 5
                    if i < length:
                        continue
                    else:
                        average = sum(lossesArray[i-length:i+1])/len(lossesArray[i-length:i+1])
                    movingAverage.append(average)

                plt.clf()
                plt.plot(np.array(epochsArray), np.array(lossesArray), linewidth=2, color='#4dbbfa', label='Loss')
                plt.plot(np.array(epochsArray[5:]), np.array(movingAverage), linewidth=2, color='#fa4d4d', label='Trend')
                plt.legend(loc='upper center')
                plt.autoscale()
                plt.draw()
                plt.pause(0.001)

            print(f'\nAverage Loss : {averageLoss}')
            print('-------------------------------\n')

            torch.save(model.state_dict(), './models/model.pt')
        plt.savefig('./loss.jpg')



    if task == 'test':
        # Test Accuracy
        #
        # [ RECORD ::: 67.50311720698254 ]
        #
        expectedOutputs = []
        predictedOutputs = []

        correct = 0
        wrong = 0

        confidenceCutoff = 0.75

        print('Starting test data . . .\n')
        for i, [input, expectedOutput] in enumerate(loader):
            input = torch.FloatTensor(input)
            prediction = model(input)[0].tolist()

            expectedOutput = expectedOutput[0].tolist()

            if expectedOutput.index(max(expectedOutput)) == prediction.index(max(prediction)) and max(prediction) / sum(prediction) >= confidenceCutoff:
                correct += 1
            elif max(prediction) / sum(prediction) >= confidenceCutoff:
                wrong += 1

        print('Accuracy :', (correct/(correct+wrong))*100, 'Out of', correct+wrong, 'trades')

    if task == 'predict':
        data = GetMostRecentTickerData('TSLA', 'current')[0]

        input = torch.FloatTensor(data)
        prediction = model(input).tolist()

        print(f'\nUp : {round( (prediction[0] / sum(prediction)) * 100 )}%\nDown : {round( (prediction[1] / sum(prediction)) * 100 )}%')
