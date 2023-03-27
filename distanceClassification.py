from data import GetTickerDataNormalized
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.concat([
    GetTickerDataNormalized('TSLA'),
    GetTickerDataNormalized('AAPL'),
    GetTickerDataNormalized('SPY'),
    GetTickerDataNormalized('AMZN')
], ignore_index=True)
print(data.to_string())

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
ax.scatter(data[x], data[y], data[z], c=color, alpha=0.1)
ax.set_xlabel(x)
ax.set_ylabel(y)
ax.set_zlabel(z)
plt.show()
