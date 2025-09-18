import pyhrv
import pyhrv.tools as tools
import datetime as dt
import warnings
import numpy as np
import json
import matplotlib.pyplot as plt
import biosppy
import os
import matplotlib.cbook
import matplotlib as mpl
from matplotlib.projections import register_projection

from matplotlib.ticker import FixedLocator, LogFormatter, ScalarFormatter
from matplotlib.scale import FuncScale

def nn_intervals(rpeaks=None):
	"""Computes the NN intervals [ms] between successive R-peaks.

	Docs:	https://pyhrv.readthedocs.io/en/latest/_pages/api/tools.html#nn-intervals-nn-intervals

	Parameter
	---------
	rpeaks : array
		R-peak times in [ms] or [s]

	Returns
	-------
	nni : array
		NN intervals in [ms]

	Raises
	------
	TypeError
		If no data provided for 'rpeaks'
	TypeError
		If data format is not list or numpy array
	TypeError
		If 'rpeaks' array contains non-integer or non-float value

	Notes
	-----
	..	You can find the documentation for this function here:
		https://pyhrv.readthedocs.io/en/latest/_pages/api/tools.html#nn-intervals-nn-intervals

	"""
	# Check input signal
	if rpeaks is None:
		raise TypeError("No data for R-peak locations provided. Please specify input data.")
	elif type(rpeaks) is not list and not np.ndarray:
		raise TypeError("List, tuple or numpy array expected, received  %s" % type(rpeaks))

	# if all(isinstance(n, int) for n in rpeaks) is False or all(isinstance(n, float) for n in rpeaks) is False:
	# 	raise TypeError("Incompatible data type in list or numpy array detected (only int or float allowed).")

	# Confirm numpy arrays & compute NN intervals
	rpeaks = np.asarray(rpeaks)
	nn_int = np.zeros(rpeaks.size - 1)

	for i in range(nn_int.size):
		nn_int[i] = rpeaks[i + 1] - rpeaks[i]

	return pyhrv.utils.nn_format(nn_int)

def heart_rate(nni=None, rpeaks=None):
	"""Computes a series of Heart Rate values in [bpm] from a series of NN intervals or R-peaks in [ms] or [s] or the HR from a single NNI.

	Docs:	https://pyhrv.readthedocs.io/en/latest/_pages/api/tools.html#heart-rate-heart-rate

	Parameters
	----------
	nni : int, float, array
		NN intervals in [ms] or [s].
	rpeaks : int, float, array
		R-peak times in [ms] or [s].

	Returns
	-------
	bpm : list, numpy array, float
		Heart rate computation [bpm].
		Float value if 1 NN interval has been provided
		Float array if series of NN intervals or R-peaks are provided.

	Raises
	------
	TypeError
		If no input data for 'rpeaks' or 'nn_intervals provided.
	TypeError
		If provided NN data is not provided in float, int, list or numpy array format.

	Notes
	-----
	..	You can find the documentation for this module here:
		https://pyhrv.readthedocs.io/en/latest/_pages/api/tools.html#heart-rate-heart-rate

	"""
	# Check input
	if nni is None and rpeaks is not None:
		# Compute NN intervals if rpeaks array is given; only 1 interval if 2 r-peaks provided
		nni = nn_intervals(rpeaks) if len(rpeaks) > 2 else int(np.abs(rpeaks[1] - rpeaks[0]))
	elif nni is not None:
		# Use given NN intervals & confirm numpy if series of NN intervals is provided
		if type(nni) is list or type(nni) is np.ndarray:
			nni = pyhrv.utils.nn_format(nni) if len(nni) > 1 else nni[0]
		elif type(nni) is int or float:
			nni = int(nni) if nni > 10 else int(nni) / 1000
	else:
		raise TypeError("No data for R-peak locations or NN intervals provided. Please specify input data.")

	# Compute heart rate data
	if type(nni) is int:
		return 60000. / float(nni)
	elif type(nni) is np.ndarray:
		return np.asarray([60000. / float(x) for x in nni])
	else:
		raise TypeError("Invalid data type. Please provide data in int, float, list or numpy array format.")
	
def heart_rate_heatplot(nni=None,
						rpeaks=None,
						signal=None,
						sampling_rate=1000.,
						age=18,
						gender='male',
						interval=None,
						figsize=None,
						show=True):
	"""Graphical visualization & classification of HR performance based on normal HR ranges by age and gender.

	Docs: https://pyhrv.readthedocs.io/en/latest/_pages/api/tools.html#heart-rate-heatplot-hr-heatplot

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	signal : array, optional
		ECG lead-I like signal.
	sampling_rate : int, float, optional
		Sampling rate of the acquired signal in [Hz].
	age : int, float
		Age of the subject (default: 18).
	gender : str
		Gender of the subject ('m', 'male', 'f', 'female'; default: 'male').
	interval : list, optional
		Sets visualization interval of the signal (default: [0, 10]).
	figsize : array, optional
		Matplotlib figure size (width, height) (default: (12, 4)).
	show : bool, optional
		If True, shows plot figure (default: True).

	Returns
	-------
	hr_heatplot : biosppy.utils.ReturnTuple object

	Raises
	------
	TypeError
		If no input data for 'nni', 'rpeaks' or 'signal' is provided

	Notes
	-----
	.. 	If both 'nni' and 'rpeaks' are provided, 'rpeaks' will be chosen over the 'nn' and the 'nni' data will be computed
		from the 'rpeaks'
	.. 	Modify the 'hr_heatplot.json' file to write own database values

	"""
	# Helper function
	def _get_classification(val, data):
		for key in data.keys():
			if data[key][0] <= int(val) <= data[key][1]:
				return key

	# Check input
	if signal is not None:
		rpeaks = biosppy.signals.ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)[2]
	elif nni is None and rpeaks is None:
		raise TypeError('No input data provided. Please specify input data.')

	# Get NNI series
	nn = pyhrv.utils.check_input(nni, rpeaks)

	# Compute HR data and
	hr_data = heart_rate(nn)
	t = np.cumsum(nn) / 1000
	interval = pyhrv.utils.check_interval(interval, limits=[0, t[-1]], default=[0, t[-1]])

	# Prepare figure
	if figsize is None:
		figsize = (12, 5)
	fig, (ax, ax1, ax2) = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [12, 1, 1]})
	ax1.axis("off")
	#fig.suptitle("Heart Rate Heat Plot (%s, %s)" % (gender, age))

	# X-Axis configuration
	# Set x-axis format to seconds if the duration of the signal <= 60s
	if interval[1] <= 60:
		ax.set_xlabel('Time [s]')
	# Set x-axis format to MM:SS if the duration of the signal > 60s and <= 1h
	elif 60 < interval[1] <= 3600:
		ax.set_xlabel('Time [MM:SS]')
		formatter = mpl.ticker.FuncFormatter(lambda ms, x: str(dt.timedelta(seconds=ms))[2:])
		ax.xaxis.set_major_formatter(formatter)
	# Set x-axis format to HH:MM:SS if the duration of the signal > 1h
	else:
		ax.set_xlabel('Time [HH:MM:SS]')
		formatter = mpl.ticker.FuncFormatter(lambda ms, x: str(dt.timedelta(seconds=ms)))
		ax.xaxis.set_major_formatter(formatter)

	# Set gender
	if gender not in ["male", "m", "female", "f"]:
		raise ValueError("Unknown gender '%s' for this database." % gender)
	else:
		if gender == 'm':
			gender = 'male'
		elif gender == 'f':
			gender = 'female'

	# Load comparison data from database
	database = json.load(open(os.path.join(os.path.split(__file__)[0], './hr_heatplot.json')))

	# Get database values
	if age > 17:
		for key in database["ages"].keys():
			if database["ages"][key][0] - 1 < age < database["ages"][key][1] + 1:
				_age = database["ages"][key][0]

		color_map = database["colors"]
		data = database[gender][str(_age)]
		order = database["order"]

		# Plot with information based on reference database:
		# Create classifier counter (preparation for steps after the plot)
		classifier_counter = {}
		for key in data.keys():
			classifier_counter[key] = 0

		# Add threshold lines based on the comparison data
		for threshold in data.keys():
			ax.hlines(data[threshold][0], 0, t[-1], linewidth=0.4, alpha=1, color=color_map[threshold])
		ax.plot(t, hr_data, 'k--', linewidth=0.5)

		# Add colorized HR markers
		old_classifier = _get_classification(hr_data[0], data)
		start_index = 0
		end_index = 0
		for hr_val in hr_data:
			classifier_counter[old_classifier] += 1
			current_classifier = _get_classification(hr_val, data)
			if current_classifier != old_classifier:
				ax.plot(t[start_index:end_index], hr_data[start_index:end_index], 'o',
						markerfacecolor=color_map[old_classifier], markeredgecolor=color_map[old_classifier])
				start_index = end_index
				old_classifier = current_classifier
			end_index += 1

		# Compute distribution of HR values in %
		percentages = {}
		_left = 0
		legend = []
		ax2.tick_params(left=False)
		ax2.set_yticklabels([])
		for i in list(range(7)):
			classifier = str(order[str(i)][0])
			percentages[classifier] = float(classifier_counter[classifier]) / hr_data.size * 100
			ax2.barh(y=0, width=percentages[classifier], left=_left, color=color_map[classifier])
			_left += percentages[classifier]
			legend.append(mpl.patches.Patch(label="%s\n(%.2f%s)" % (order[str(i)][1], percentages[classifier], "$\%$"),
											fc=color_map[classifier]))
		ax.legend(handles=legend, loc=8, ncol=7)
	elif age <= 0:
		raise ValueError("Age cannot be <= 0.")
	else:
		warnings.warn("No reference data for age %i available." % age)
		ax.plot(t, hr_data, 'k--', linewidth=0.5)
		ax2.plot("", 0)

	# Set axis limits
	ax.axis([interval[0], interval[1], hr_data.min() * 0.7, hr_data.max() * 1.1])
	ax.set_ylabel('Heart Rate [$1/min$]')
	ax2.set_xlim([0, 100])
	ax2.set_xlabel("Distribution of HR over the HR classifiers [$\%$]")

	# Show plot
	if show:
		plt.show()

	# Output
	return biosppy.utils.ReturnTuple((fig, ), ('hr_heatplot', ))
