import yfinance as yf
import pandas_ta as pta
import pandas
import numpy as np

def GetTickerData(ticker):
	data = yf.download(tickers=ticker, period="60d", interval = "5m", ignore_tz = False, prepost = False, progress = False)

	spread = [0, 0, 0]

	inputs = []
	expectedOutputs = []

	bp = 10
	fcc = 30

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
	for i in range(100, len(data['Close'])-32):
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

		for j in range(1, 31):
			startValue = data['Close'][i]
			longPosition = startValue * 1.005
			shortPosition = startValue * 0.995
			if data['High'][i+j] > longPosition:
				expectedOutputs.append([0, 0, 1])
				spread[2] += 1
				break
			elif data['Low'][i+j] < shortPosition:
				expectedOutputs.append([1, 0, 0])
				spread[0] += 1
				break
		else:
			expectedOutputs.append([0, 1, 0])
			spread[1] += 1

	print(f'{ticker}')
	return [inputs, expectedOutputs]
