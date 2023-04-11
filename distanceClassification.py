from data import GetTickerDataNormalized
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from math import sqrt

tsla = GetTickerDataNormalized('TSLA')
lastValue = pd.DataFrame([tsla.reset_index(drop=True).values.tolist()[-1]], columns=tsla.columns)

data = pd.concat([
    pd.DataFrame(tsla.dropna().reset_index(drop=True).values.tolist()[:len(tsla['RSI']-2)], columns=tsla.columns),
], ignore_index=True)
print(lastValue.to_string())

values = []
weights = []
for di in range(len(data['Change'])-1):
    changeDistance = (lastValue['Change'] - data['Change'][di]) / 2
    rsiDistance = lastValue['RSI'] - data['RSI'][di]
    choppinessDistance = lastValue['Choppiness'] - data['Choppiness'][di]
    adxDistance = lastValue['ADX'] - data['ADX'][di]
    willrDistance = lastValue['Williams %R'] - data['Williams %R'][di]
    diffDistance = (lastValue['Difference From EMA'] - data['Difference From EMA'][di]) / 5
    stochKDistance = lastValue['Stochastic K'] - data['Stochastic K'][di]
    stochDDistance = lastValue['Stochastic D'] - data['Stochastic D'][di]
    aroonUDistance = lastValue['Aroon Up'] - data['Aroon Up'][di]
    aroonDDistance = lastValue['Aroon Down'] - data['Aroon Down'][di]

    spatialDistance = sqrt(changeDistance**2 + rsiDistance**2 + choppinessDistance**2 + adxDistance**2 + willrDistance**2 + diffDistance**2 + stochDDistance**2 + stochKDistance**2 + aroonUDistance**2 + aroonDDistance**2)
    weights.append(0.1 / (30 * spatialDistance - 3))
    values.append(data['Target'][di])

totalSum = 0
for di in range(len(data['Change'])-1):
    totalSum += values[di] * (weights[di])
totalSum /= sum(weights)
print(totalSum)


plot = False
if plot:
    color = []
    for target in data['Target']:
        if target == 1:
           color.append((0, 1, 0))
        elif target == 0:
           color.append((0.2, 0.2, 0.2))
        else:
           color.append((1, 0, 0))

    x = 'Change'
    y = 'Difference From EMA'
    z = 'RSI'
    plt.style.use('ggplot')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(data[x], data[y], data[z], c=color, alpha=0.7)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()
