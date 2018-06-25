"""
	The graph maker can conveniently be used to create graphs by adding data in multiple steps of your program.
	It can be stored and read automatically and 
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import numpy as np
import matplotlib
import pickle
from .misc_func import *
import itertools

class Graph_Maker():
	"""
		Class used to collect x and y values and then plot it all at once. 
		It should be stored everytime, this is useful in case you want to collect data from multiple
		points in the program, and are unsure of how it should be plotted. 
	"""

	def __init__(self, load=False, filename="graph_maker/obj1", fontsize=15):
		self.x = []
		self.y = []
		self.colors = ['r', 'b', 'g', 'y', 'black', 'cyan', 'magenta', 'orange']
		matplotlib.rcParams.update({'font.size': fontsize})
		if load:
			self.load(filename=filename)

	def add_x(self, x, if_exists=True):
		"""
			Set the new x-axis of the graph, the boolean should be set to false if you dont want to overwrite it if
			it already exists.
		"""
		if if_exists:
			self.x = x

	def add_y(self, y, if_exists=True):
		"""
			Set the new y-axis of the graph, the boolean should be set to false if you dont want to overwrite it if
			it already exists.
		"""
		if if_exists:
			self.y = y

	def append_y(self, y):
		"""
			add another y value
		"""
		self.y.append(y)

	def append_x(self, x):
		"""
			add another y value
		"""
		self.x.append(x)

	def subplot_n(self, xlabel, ylabel, main_title, subtitles, labels, xticks=None):
		"""
			Plot n subplots. These are all line graphs. 
		"""
		num_subplots = len(self.x) # The number of lists in the x variable = number of subplots
		num_ys = len(self.y) / num_subplots
		fig, axes = plt.subplots(num_subplots, sharex=True, sharey=True)
		fig.suptitle(main_title)
		i = 0
		for ax in axes:
			subtitle = subtitles[i]
			ax.set_title(subtitle)
			x = self.x[i]
			for f in range(0, int(num_ys)):
				color = self.colors[f]
				label = labels.pop(0)
				y = self.y.pop(0)
				ax.plot(x, y, label=label, color=color)
				ax.set_xticks(x)
				if xticks != None:
					ax.set_xticklabels(xticks)
			ax.set_ylabel(ylabel)
			i += 1
		plt.xlabel(xlabel)
		plt.legend(bbox_to_anchor=(1.1, 2.05))
		plt.show()

	def plot_line_n(self, xlabel, ylabel, title, labels, subtitle=None, xticks=None):
		"""
			Plot n lines in a single figure
		"""
		colors=self.colors
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		if subtitle != None:
			plt.suptitle(subtitle, fontsize=12)

		for i in range(0, len(self.y)):
			y = self.y[i]
			label = labels[i]
			color = colors[i]
			plt.plot(self.x, y, color=color, label=label, alpha=0.5)
		if xticks != None:
			plt.xticks(self.x, xticks)
		plt.tight_layout()
		plt.legend()
		plt.show()


	def plot_bar(self, xlabel, ylabel, title, xticks_x=None, xticks_label=None, subtitle=None, show_value=False, rotation=45):
		"""
			Create a bar graph of the data
		"""
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if xticks_x != None:
			plt.xticks(xticks_x, xticks_label, fontsize=10, rotation=rotation)
		plt.title(title)
		if subtitle != None:
			plt.suptitle(subtitle)
		plt.xticks(rotation=rotation)
		plt.bar(self.x, self.y, color='r', alpha=0.5)
		
		if show_value:
			for x in range(0, len(self.x)):
				plt.text(x, self.y[x]/2.0, str(self.y[x]))
		plt.tight_layout()
		plt.show()

	def plot_bar_enlarged(self, xlabel, ylabel, title, xticks_x=None, \
		xticks_label=None, subxax=None):
		"""
			Make a bar graph with a section of the bar graph highlighted, requires some manual fiddeling
		"""
		xdata = self.x
		ydata = self.y
		fig, ax = plt.subplots() # create a new figure with a default 111 subplot
		ax.bar(xdata, ydata, color='g')
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		if xticks_x != None:
			plt.xticks(xticks_x, xticks_label, fontsize=10, rotation=45)

		# Smaller subsection
		desired_portion = 0.5
		zoom = desired_portion / (len(subxax) / len(ydata))
		axins = zoomed_inset_axes(ax, zoom, loc=5)
		subydata = ydata[0:len(subxax)]
		height = 0.02
		newydata = np.array(subydata) * (height / max(subydata))
		axins.bar(subxax, newydata)
		plt.yticks(visible=False)
		axins.set_xlabel("Characters")
		axins.set_title("Subplot of character distribution")
		mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
		plt.tight_layout()
		plt.show()

	def plot_scatter(self, xlabel, ylabel, title, avg=False, text=None):
		"""
			Scatter plot from the data
		"""
		xdata = self.x
		ydata = self.y
		fig = plt.figure(dpi=200, figsize=(10, 5))
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		
		plt.title(title)
		plt.scatter(xdata, ydata, label="Scatter Accuracy")
		if avg:
			# Also plot the average scatter value
			x, y = zip(*sorted((xVal, np.mean([yVal for a, yVal in zip(xdata, ydata) if xVal==a])) for xVal in set(xdata)))
			plt.plot(x, y, label="Average Accuracy", color='r')
		plt.text(7, 0.6, text, fontsize=10)
		plt.plot()
		plt.tight_layout()
		axes = plt.gca()
		axes.set_ylim([0.5,1])
		plt.legend()
		plt.show()

	def plot_bar_double(self, xlabel, ylabel, title, label1, label2):
		"""
			Plot two bar graphs over one another
		"""
		xdata = self.x
		ydata1 = self.y[0]
		ydata2 = self.y[1]
		fig = plt.figure(dpi=200, figsize=(10, 5))
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.xticks(rotation=90)
		
		indices = np.arange(len(xdata))
		plt.title(title)
		plt.bar(indices, ydata1, color='b', label=label1, alpha=1)
		plt.bar(indices, ydata2, width=0.5, color='r', alpha=1, label=label2)
		plt.xticks(indices, xdata)
		plt.tight_layout()
		plt.legend()
		plt.show()

	def plot_bar_n(self, xlabel, ylabel, title, labels, subtitle=None, rotation=45):
		"""
			Plot n bars per class
		"""
		colors=self.colors
		width = 0.9 / (float(len(self.y))) # always leave a little room
		sep = width / float(len(self.y))
		xdata = self.x
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.xticks(rotation=rotation)
		if subtitle != None:
			plt.suptitle(subtitle, fontsize=12)
		indices = np.arange(len(xdata))
		plt.title(title)
		plot_indices = []
		minimal = -len(self.y) /2.0 * width
		for i in range(0, len(self.y)):
			tmp_sep = minimal + (i * width)
			y = self.y[i]
			label = labels[i]
			color = colors[i]
			plt.bar(indices - tmp_sep, y, width=width, color=color, label=label, alpha=0.5)
			r = 0
			for x in (indices - tmp_sep):
				plt.text(x, y[r]/2.0, str(y[r]), horizontalalignment="center")
				r += 1
		
		plt.xticks(indices, xdata)
		plt.tight_layout()
		plt.legend()
		plt.show()

	def plot_bar_double_scale(self, xlabel, ylabel1, ylabel2, title, label1, label2):
		"""
			Plot 2 bar graphs but with a double scale
		"""
		fig = plt.figure()

		indices = np.arange(len(self.x))

		width = 0.4
		sep = width / 2
		ax1 = fig.add_subplot(111)
		p1 = ax1.bar(indices+sep, self.y[0], width=width, color='b', label=label1, alpha=1)
		ax1.set_ylabel(ylabel1)
		plt.xticks(rotation=70)

		ax2 = ax1.twinx()
		p2 = ax2.bar(indices-sep, self.y[1], width=width, color='r', alpha=1, label=label2)
		ax2.set_ylabel(ylabel2, color='r')
		for tl in ax2.get_yticklabels():
			tl.set_color('r')
		ax1.set_xlabel(xlabel)
		plt.xticks(indices, self.x)
		plt.legend(handles=[p1, p2])
		plt.title(title)
		plt.tight_layout()
		plt.show()

	def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
			"""
			This function prints and plots the confusion matrix.
			Normalization can be applied by setting `normalize=True`.
			"""
			if normalize:
				cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
				print("Normalized confusion matrix")
			else:
				print('Confusion matrix, without normalization')

			plt.imshow(cm, interpolation='nearest', cmap=cmap)
			plt.title(title)
			plt.colorbar()
			tick_marks = np.arange(len(classes))
			plt.xticks(tick_marks, classes, rotation=90)
			plt.yticks(tick_marks, classes)

			fmt = '.2f' if normalize else 'd'
			thresh = cm.max() / 2.
			for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
				plt.text(j, i, format(cm[i, j], fmt),
						 horizontalalignment="center",
						 color="white" if cm[i, j] > thresh else "black", fontsize=7)

			plt.tight_layout()
			plt.ylabel('True label')
			plt.xlabel('Predicted label')
			plt.show()

	def plot_confusion_small_font(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		for i in range(0, len(classes)):
			if len(classes[i]) > 30:
				classes[i] = classes[i][-30::]
		plt.xticks(tick_marks, classes, rotation=90)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black", fontsize=7)
		# plt.tick_params(labelsize=8)
		# plt.gcf().subplots_adjust(bottom=0.9, top=10)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()
		
	def store(self, filename="/graph_maker/obj1"):
		"""
			Store the object
		"""
		import os 
		dir_path = os.path.dirname(os.path.realpath(__file__))
		filename = dir_path + filename
		store_file(filename, self)

	def load(self, filename="/graph_maker/obj1"):
		"""
			Load the configuration of a previously stored object
		"""
		import os 
		dir_path = os.path.dirname(os.path.realpath(__file__))
		filename = dir_path + filename
		gm = read_file(filename)
		self.x = gm.x
		self.y = gm.y

	def __repr__(self):
		print("x: ", str(self.x))
		if self.y != []:
			if type(self.y[0]) == list:
				print("y values: ")
				print_list(self.y)
			else:
				print("y: ", str(self.y))
		else:
			print("y: ", str(self.y))
		return ""
