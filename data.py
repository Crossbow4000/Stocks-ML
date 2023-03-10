import yfinance as yf
import pandas_ta as pta
import numpy as np

def GetTickerData(ticker):
	data = yf.download(tickers = ticker, period = "60d", interval = "5m", ignore_tz = False, prepost = False, progress = False)

	inputs = []
	expectedOutputs = []

	dataRSI = pta.rsi(data['Close'], 14)
	dataSTOCH = pta.stoch(data['High'], data['Low'], data['Close'])
	dataMACD = pta.macd(data['Close'], 12, 26)
	dataCHOP = pta.chop(data['High'], data['Low'], data['Close'])
	dataTSI = pta.tsi(data['Close'], fast=13, slow=25, signal=13)
	dataADX = pta.adx(data['High'], data['Low'], data['Close'])
	for i in range(50, len(data['Close'])-21):
		close = np.array(data['Close'][i-10:i])
		rsi = np.array(dataRSI[i-10:i])
		stochK = np.array(dataSTOCH[i-10:i]['STOCHk_14_3_3'])
		stochD = np.array(dataSTOCH[i-10:i]['STOCHd_14_3_3'])
		macd = np.array(dataMACD[i-10:i]['MACD_12_26_9'])
		macdHistogram = np.array(dataMACD[i-10:i]['MACDh_12_26_9'])
		macdSignal = np.array(dataMACD[i-10:i]['MACDs_12_26_9'])
		chop = np.array(dataCHOP[i-10:i])
		tsi = np.array(dataTSI[i-10:i]['TSI_13_25_13'])
		tsiS = np.array(dataTSI[i-10:i]['TSIs_13_25_13'])
		adx = np.array(dataADX[i-10:i]['ADX_14'])
		dmp = np.array(dataADX[i-10:i]['DMP_14'])
		dmn = np.array(dataADX[i-10:i]['DMN_14'])

		input = []
		input.extend(rsi.tolist())
		input.extend(stochK.tolist())
		input.extend(stochD.tolist())
		input.extend(macd.tolist())
		input.extend(macdHistogram.tolist())
		input.extend(macdSignal.tolist())
		input.extend(chop.tolist())
		input.extend(tsi.tolist())
		input.extend(tsiS.tolist())
		input.extend(adx.tolist())
		input.extend(dmp.tolist())
		input.extend(dmn.tolist())
		inputs.append(input)

		for j in range(0, 20):
			if ((data['Close'][i+j] - close[9]) / close[9]) * 100 > 0.5:
				expectedOutputs.append([1])
				break
			elif ((data['Close'][i+j] - close[9]) / close[9]) * 100 < -0.5:
				expectedOutputs.append([-1])
				break
			else:
				expectedOutputs.append([0])
				break

	return [inputs, expectedOutputs]
