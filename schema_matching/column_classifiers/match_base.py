
from ..misc_func import *
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from abc import ABCMeta, abstractmethod
import operator
import numpy as np
import sys

class Matcher(metaclass=ABCMeta):
	"""
		A matcher can have multiple features. Each feature should contain an extract_features method, which 
		extracts features based on column data. 

		Each matcher should also have a classifying element (called clf here). It can be overwritten per matcher.
		It should be trained upon initiation
	"""

	def __init__(self, featuremap):
		self.featuremap = featuremap
		svm = LinearSVC(class_weight = 'balanced')
		self.clf = CalibratedClassifierCV(svm)
		self.outlier_detector_dict = {}

	@abstractmethod
	def train(self):
		"""
			Should be called upon initiation and handle the training of the matchers, as well as the 
			outlier detectors
		"""
		raise NotImplementedError

	@abstractmethod
	def classify_column(self):
		"""
			Classify a column based on inputted column. 
			Should call the .extract_features_column(column) method from the features to find the datapoints. 
			
			returns ['prediction'] or returns 'classname'
			
			If outlier has to be detected:
			return ['prediction'], [[a, b]] (depending on whether or not it is an outlier)

			a is the chance of the input to be an outlier, and b is the chance of it to be an inlier
		"""
		raise NotImplementedError

	@abstractmethod
	def classify_instance(self):
		"""
			Classify an instance based on the inputted entry.

			returns ['prediction']
		"""
		raise NotImplementedError

	@abstractmethod
	def classify_column_proba(self):
		"""
			Returns the probabilities for each class. Make sure if you use this method that the get_classes() method
			returns the classes in the correct order. 

			Returns [[probabilities per class]] [[a, b]] (depending on whether or not it is an outlier)
			a is the chance of the input to be an outlier, and b is the chance of it to be an inlier
		"""
		raise NotImplementedError

	@abstractmethod
	def classify_instance_proba(self):
		"""
			Returns probability for each class for an instance. Make sure if you use this method that the get_classes() method
			returns the classes in the correct order.

			Returns [[Probabilities per class]]
		"""
		raise NotImplementedError

	def create_oneclass_dict(self, datapoints, targetpoints, contamination=0.5):
		"""
			For each class we train a binary classifier to check if the datapoints actually belong to the class
			itself. We do this by introducing noise, a.k.a data from other classes. 
		"""
		all_classes = list(set(targetpoints))
		for entity in all_classes:
			dataset, targets, contamination_rate = self.create_oneclass_dataset(datapoints, targetpoints, entity, contamination)
			clf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
			clf.fit(dataset, targets)
			self.outlier_detector_dict[entity] = clf

	def create_oneclass_dataset(self, datapoints, targetpoints, main_class, contamination=0.5):
		"""
			Create 1 set of datapoints with noise introduced, based on the contamination rate.
			First serperate the data, and then randomly introduce noise points
		"""
		main_class_data = []
		noise = []
		targets = []
		for i in range(0, len(datapoints)):
			target_class = targetpoints[i]
			datapoint = datapoints[i]
			if target_class == main_class:
				main_class_data.append(datapoint)
				targets.append(1)
			else:
				noise.append(datapoint)
		num_noise = int(len(main_class_data) / (1-contamination) - len(main_class_data))
		if num_noise + len(main_class_data) > len(datapoints):
			""" If there arent enough examples just add as many as possible """
			num_noise = len(noise)
		np.random.shuffle(noise)
		main_class_data += noise[0:num_noise]
		targets += np.zeros(num_noise).tolist()
		return main_class_data, targets, float(num_noise)/len(main_class_data)
		

	def most_common(self, myList):
		""" Get the most common item in a list """
		result = {}
		for elem in myList:
			if elem in result:
				result[elem] += 1
			else:
				result[elem] = 1
		x = max(result.items(), key=operator.itemgetter(1))[0]
		return x

	def get_classes(self):
		return self.clf.classes_
