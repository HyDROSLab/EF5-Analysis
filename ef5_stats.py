#!/usr/bin/env python
import csv
import sys
import datetime as DT
import numpy as np
from matplotlib.dates import date2num, num2date
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def format_date(x, pos=None):
        thisind = np.clip(int(startInd + x + 0.5), startInd, startInd + numTimes - 1)
        return num2date(times2[thisind]).strftime('%m/%d/%Y %H:%M')

def parse_date(x):
	return date2num(DT.datetime.strptime(x, "%Y-%m-%d %H:%M"))

def make_stats_plot(filename_in, plotTitle, startTime=0, endTime=0, forecastTime=0):
	global startInd
	global numTimes
	global times2
	dtype = [('times', 'S16')] + [('', np.float32)]*4

	with open(filename_in) as f:
        	y = np.loadtxt(f, delimiter=',', dtype=dtype, skiprows=1)

	y = y.view(np.dtype([('times', 'S16'), ('data', np.float32, 4)]))
	data = y['data']
	times = [s.astype(str) for (s) in y['times']]
	sim = data[:,0]
	obs = data[:,1]
	precip = data[:,2]
	pet = data[:,3]
	rain = precip #- pet
	times2 = [date2num(DT.datetime.strptime(s, "%Y-%m-%d %H:%M")) for (s) in times]
	if startTime == 0:
        	startInd = 0
        	numTimes = len(times2)
        	endInd = numTimes
	else:
        	startInd = 0
        	while times2[startInd] < startTime:
                	startInd = startInd + 1
        	endInd = len(times2) - 1
        	while times2[endInd] > endTime:
                	endInd = endInd - 1
        	numTimes = endInd - startInd + 1
        	endInd = endInd + 1
	obs2 = np.array([obs[x] for x in range(0, len(obs)) if obs[x] == obs[x] and x >= startInd and x <= endInd])
	sim2 = np.array([sim[x] for x in range(0, len(sim)) if obs[x]==obs[x] and x >= startInd and x <= endInd])
	times_obs = np.array([times[x] for x in range(0, len(times)) if obs[x]==obs[x] and x >= startInd and x <= endInd])
	indObsReal = np.array([x for x in range(0, len(times)) if obs[x]==obs[x] and x >= startInd and x <= endInd])
	times_obs2 = [date2num(DT.datetime.strptime(s, "%Y-%m-%d %H:%M")) for (s) in times_obs]
	#print('Rain sum:' + str(precip.sum() / 12.0))

	#CC
	if len(obs2) == 0:
		meansim = 0.0
		meanobs = 0.0
		meanprecip = 0.0
		bias = 0.0
		nsce = 0.0
		CC = 0.0
		modCC = 0.0
		mae = 0.0
		rmse = 0.0
		maxdiff = 0.0
		maxdifftime = 0.0
	else:
		meansim = sim2.mean()
		meanobs = obs2.mean()
		meanprecip = rain.mean()
		bias = (meansim/meanobs - 1.0) * 100.0 #Bias in %
		#print(meanobs)
        	# NSCE calculation
		num = np.sum((sim2 - obs2)**2.0)
		den = np.sum((obs2 - meanobs)**2.0)
		nsce = 1.0 - num/den
		stdobs = obs2.std()
		stdsim = sim2.std()
		CCm = np.cov(obs2, y=sim2)
		CC = CCm[0][1] / (stdobs * stdsim);

		#modCC
		if (stdsim > stdobs):
			modCC = CC * (stdobs/stdsim) 
		else:
			modCC = CC * (stdsim/stdobs)

		#MAE
		mae = np.sum(np.abs(sim2 - obs2)) / obs2.size

		#RMSE
		rmse = np.sqrt(np.sum((sim2 - obs2)**2.0) / obs2.size)

		#max diff
		maxdiff = sim2.max() - obs2.max()

		#max diff time
		maxdifftime = obs2.argmax() - sim2.argmax()

		#print('NSCE: ' + str(nsce))
		#print('Bias: ' + str(bias))
		#print('CC: ' + str(CC))
		#print('modCC: ' + str(modCC))
		#print('MAE: ' + str(mae))
		#print('RMSE: ' + str(rmse))
		#print('peakerror: ' + str(maxdiff))
		#print('peak: ' + str(sim2.mean()))
		#print('tpeakerror: ' + str(maxdifftime))
		#print('Mean Precip: ' + str(meanprecip))

	if startTime == 0:
		startInd = 0
		numTimes = len(times2)
		endInd = numTimes
		startIndObs = 0
		numTimesObs = len(times_obs2)
		endIndObs = numTimesObs
	else:
		startInd = 0
		while times2[startInd] < startTime:
			startInd = startInd + 1
		endInd = len(times2) - 1
		while times2[endInd] > endTime:
			endInd = endInd - 1
		numTimes = endInd - startInd + 1
		endInd = endInd + 1

		startIndObs = 0
		numTimesObs = len(times_obs2)
		endIndObs = numTimesObs

	indObs = np.array([indObsReal[x] - indObsReal[startIndObs] for x in range(startIndObs, endIndObs)])

	font = {'family' : 'sans-serif',
		'weight' : 'bold',
		'size'   : 22}
	matplotlib.rc('font', **font)

	fig = plt.figure(figsize=[22, 12], dpi=120, facecolor='0.9')
	fig.subplots_adjust(top=0.85,right=0.92)
	if len(obs2) > 0:
		dataMax = max(obs2[startIndObs:endIndObs].max()*1.25, sim[startInd:endInd].max()*1.25)
	else:
		dataMax = sim[startInd:endInd].max()*1.25
	#print("Data Max")
	#print(dataMax)
	precipMax = rain[startInd:endInd].max() * 3.0
	ind = np.arange(numTimes)
	#indObs = np.arange(numTimesObs)
	ax = fig.add_subplot(111)
	ax.yaxis.grid(True, color='black', linestyle='dashed')
	ax.xaxis.grid(True, color='black', linestyle='dashed')
	ax.set_axisbelow(True)
	ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
	fig.autofmt_xdate()
	if len(obs) > 0:
		#ax.fill_between(ind, 0, mObs_masked, facecolor='black', alpha=0.5)
		obsp, = ax.plot(ind, obs[startInd:endInd], 'ko')
	simp, = ax.plot(ind, sim[startInd:endInd], linewidth=3, color='blue', alpha=0.75)
	#ax.set_xticks(np.arange(numTicks + 1) * 24)
	ax.set_xlim(0, numTimes)
	ax.set_ylim(0, dataMax)
	ax.set_xlabel('Time (UTC)')
	ax.set_ylabel('Discharge (cms)')
	ax2 = plt.twinx()
	ax2.set_xlim(0, numTimes)
	ax2.set_ylim(precipMax, 0)
	#ax2.set_ylabel('Return Period (years)')
	ax2.set_ylabel('Basin Avg Rainfall (mm/h)')
	ax2.fill_between(ind, 0, rain[startInd:endInd], facecolor='green', alpha=0.7)
	precipp = ax2.plot(ind, rain[startInd:endInd], linewidth=3, color='green', alpha=0.75)
	if forecastTime > 0:
		forecastX = 0
		while times2[forecastX] <= forecastTime:
			forecastX = forecastX + 1
		forecastX = forecastX - startInd
		ax.axvline(x=forecastX, linewidth=2, color='burlywood')
	#ax2.axhline(y=2.0, linewidth=2, color='darksalmon', linestyle='--')
	ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
	fig.autofmt_xdate()
	if len(obs2) > 0:
		plt.legend([obsp, simp], ["Observed", "Simulated"], loc='upper right', bbox_to_anchor=(1, 1.22), fancybox=True, shadow=True)
		fig.text(0.01, .97, "NSCE: %.2f" % nsce)
		fig.text(0.01, .94, "Bias: %.2f" % bias)
		fig.text(0.01, .91, "CC: %.2f" % CC)
		fig.text(0.01, .88, "ModCC: %.2f" % modCC)
		fig.text(0.21, .97, "MAE: %.2f" % mae)
		fig.text(0.21, .94, "RMSE: %.2f" % rmse)
		fig.text(0.21, .91, "Peak ERR: %.2f" % maxdiff)
		fig.text(0.21, .88, "Peak Time ERR: %.2f" % maxdifftime)
	else:
		plt.legend([simp], ["Simulated"], loc='upper right', bbox_to_anchor=(1, 1.22), fancybox=True, shadow=True)
	fig.text(0.01, 0.01, "EF5-Stats: %s" % filename_in, size=12)
	plt.title(plotTitle)
	plt.show()

if __name__ == '__main__':
	if (len(sys.argv) < 2):
        	print('You must supply the CSV file name to produce stats for!')
        	sys.exit(1)


	title = ""
	if (len(sys.argv) == 5):
		startTime = date2num(DT.datetime.strptime(sys.argv[2], "%m/%d/%Y %H:%M"))
		endTime = date2num(DT.datetime.strptime(sys.argv[3], "%m/%d/%Y %H:%M"))
		print(endTime)
		title = sys.argv[4]
	else:
		startTime = 0
		endTime = 0
		title = sys.argv[2]
	outname = sys.argv[1] + '.png'
	make_stats_plot(sys.argv[1], outname, startTime, endTime, 0, title)	
