import yfinance as yf
import pandas_ta as pta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def GetTickerData(ticker):
	data = yf.download(tickers=ticker, period="10y", interval = "1d", ignore_tz = False, prepost = False, progress = False)

	spread = [0, 0, 0]

	inputs = []
	expectedOutputs = []

	bp = 10

	atr = pta.atr(data['High'], data['Low'], data['Close'], len(data['Close'])-1)[-1]

	dataChange = data['Close'].pct_change()
	dataRSI = pta.rsi(data['Close'], 14)
	dataSTOCH = pta.stoch(data['High'], data['Low'], data['Close'])
	dataCHOP = pta.chop(data['High'], data['Low'], data['Close'])
	dataADX = pta.adx(data['High'], data['Low'], data['Close'])
	dataAROON = pta.aroon(data['High'], data['Low'], 14)
	dataBOP = pta.bop(data['Open'], data['High'], data['Low'], data['Close'])
	dataCCI = pta.cci(data['High'], data['Low'], data['Close'])
	dataWILLR = pta.willr(data['High'], data['Low'], data['Close'])
	dataEMA = (data['Close'] > pta.ema(data['Close'], 50)).astype(int)
	for i in range(100, len(data['Close'])-10):
		change = np.array(dataChange[i-bp:i]) * 100
		rsi = np.array(dataRSI[i-bp:i]) / 100
		stochK = np.array(dataSTOCH[i-bp:i]['STOCHk_14_3_3']) / 100
		stochD = np.array(dataSTOCH[i-bp:i]['STOCHd_14_3_3']) / 100
		chop = np.array(dataCHOP[i-bp:i]) / 100
		adx = np.array(dataADX[i-bp:i]['ADX_14']) / 100
		aroonUp = np.array(dataAROON[i-bp:i]['AROONU_14']) / 100
		aroonDown = np.array(dataAROON[i-bp:i]['AROOND_14']) / 100
		bop = (np.array(dataBOP[i-bp:i]) + 1)
		cci = np.array(dataCCI[i-bp:i]) / 100
		willr = np.array(dataWILLR[i-bp:i]) / -100
		ema = np.array(dataEMA[i-bp:i])

		input = []
		input.extend(change.tolist())
		input.extend(rsi.tolist())
		input.extend(stochK.tolist())
		input.extend(stochD.tolist())
		input.extend(chop.tolist())
		input.extend(adx.tolist())
		input.extend(aroonUp.tolist())
		input.extend(aroonDown.tolist())
		input.extend(bop.tolist())
		input.extend(cci.tolist())
		input.extend(willr.tolist())
		input.extend(ema.tolist())
		inputs.append(input)
		if len(inputs[-1]) != 120:
			inputs.pop()
		else:
			if dataChange[i+1] > 0:
				expectedOutputs.append([0, 1])
				spread[1] += 1
			else:
				expectedOutputs.append([1, 0])
				spread[0] += 1

	print(f'{ticker}\nUp : {spread[1]/sum(spread)}\nDown : {spread[0]/sum(spread)}\n')
	return [inputs, expectedOutputs]

def GetMostRecentTickerData(ticker, currentOrNext='current'):
	data = yf.download(tickers=ticker, period="60d", interval = "1d", ignore_tz = False, prepost = False, progress = False)

	bp = 10

	dataChange = data['Close'].pct_change()
	dataRSI = pta.rsi(data['Close'], 14)
	dataSTOCH = pta.stoch(data['High'], data['Low'], data['Close'])
	dataCHOP = pta.chop(data['High'], data['Low'], data['Close'])
	dataADX = pta.adx(data['High'], data['Low'], data['Close'])
	dataAROON = pta.aroon(data['High'], data['Low'], 14)
	dataBOP = pta.bop(data['Open'], data['High'], data['Low'], data['Close'])
	dataCCI = pta.cci(data['High'], data['Low'], data['Close'])
	dataWILLR = pta.willr(data['High'], data['Low'], data['Close'])
	dataEMA = (data['Close'] > pta.ema(data['Close'], 50)).astype(int)
	
	if currentOrNext == 'current':
		change = np.array(dataChange[-11:-1]) * 100
		rsi = np.array(dataRSI[-11:-1]) / 100
		stochK = np.array(dataSTOCH[-11:-1]['STOCHk_14_3_3']) / 100
		stochD = np.array(dataSTOCH[-11:-1]['STOCHd_14_3_3']) / 100
		chop = np.array(dataCHOP[-11:-1]) / 100
		adx = np.array(dataADX[-11:-1]['ADX_14']) / 100
		aroonUp = np.array(dataAROON[-11:-1]['AROONU_14']) / 100
		aroonDown = np.array(dataAROON[-11:-1]['AROOND_14']) / 100
		bop = (np.array(dataBOP[-11:-1]) + 1)
		cci = np.array(dataCCI[-11:-1]) / 100
		willr = np.array(dataWILLR[-11:-1]) / -100
		ema = np.array(dataEMA[-11:-1])
	else:
		change = np.array(dataChange[-10:]) * 100
		rsi = np.array(dataRSI[-10:]) / 100
		stochK = np.array(dataSTOCH[-10:]['STOCHk_14_3_3']) / 100
		stochD = np.array(dataSTOCH[-10:]['STOCHd_14_3_3']) / 100
		chop = np.array(dataCHOP[-10:]) / 100
		adx = np.array(dataADX[-10:]['ADX_14']) / 100
		aroonUp = np.array(dataAROON[-10:]['AROONU_14']) / 100
		aroonDown = np.array(dataAROON[-10:]['AROOND_14']) / 100
		bop = (np.array(dataBOP[-10:]) + 1)
		cci = np.array(dataCCI[-10:]) / 100
		willr = np.array(dataWILLR[-10:]) / -100
		ema = np.array(dataEMA[-10:])

	input = []
	input.extend(change.tolist())
	input.extend(rsi.tolist())
	input.extend(stochK.tolist())
	input.extend(stochD.tolist())
	input.extend(chop.tolist())
	input.extend(adx.tolist())
	input.extend(aroonUp.tolist())
	input.extend(aroonDown.tolist())
	input.extend(bop.tolist())
	input.extend(cci.tolist())
	input.extend(willr.tolist())
	input.extend(ema.tolist())
	return [input]

GetMostRecentTickerData('TSLA')


def GetTickerDataNormalized(ticker):
	data = yf.download(tickers=ticker, period="60d", interval = "5m", ignore_tz = False, prepost = False, progress = False)
	scaler = MinMaxScaler()

	open = data['Open']
	high = data['Close']
	low = data['Low']
	close = data['Close']

	data = pd.DataFrame()

	data['Change'] = close.pct_change() * 100
	data['RSI'] = pta.rsi(close, 14) / 100
	data['Choppiness'] = pta.chop(high, low, close) / 100
	data['ADX'] = pta.adx(high, low, close)['ADX_14'] / 100
	data['Williams %R'] = pta.willr(high, low, close) / -100
	data['Difference From EMA'] = ((close - pta.ema(close, 50)) / close) * 100

	stochastic = pta.stoch(high, low, close)
	data['Stochastic K'] = stochastic['STOCHk_14_3_3'] / 100
	data['Stochastic D'] = stochastic['STOCHd_14_3_3'] / 100

	aroon = pta.aroon(high, low)
	data['Aroon Up'] = aroon['AROONU_14'] / 100
	data['Aroon Down'] = aroon['AROOND_14'] / 100

	target = []
	for i in range(len(data['Change'])):
		for j in range(0, 17):
			change = sum(data['Change'][i+1:i+j])
			if change >= 1:
				target.append(1)
				break
			elif change <= -1:
				target.append(-1)
				break
		else:
			target.append(0)
	data['Target'] = target

	return data
