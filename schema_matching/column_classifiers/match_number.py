

from .match_base import *
import numpy as np
from .tester import Tester

class Number_Matcher(Matcher, Tester):

	def __init__(self, featuremap):
		Matcher.__init__(self, featuremap)
		self.outlier_detector = IsolationForest()
		self.train()

	def train(self):
		number_feature = self.featuremap['number_feature']
		datapoints, targetpoints = number_feature.get_features_targets()
		self.outlier_detector.fit(datapoints)
		self.create_oneclass_dict(datapoints, targetpoints)
		self.train_manual(datapoints, targetpoints)

	def train_manual(self, datapoints, targetpoints):
		self.clf.fit(datapoints, targetpoints)

	def classify_column(self, column, detect_outlier=False):
		""" 
			The column should be a list with strings to classify.
			Here we can create our own custom algorithm to match a column. 
		"""
		nf = self.featuremap['number_feature']
		datapoints = nf.extract_features_column(column)
		result = self.clf.predict(datapoints)
		if detect_outlier:
			outlier = self.outlier_detector_dict[result].predict(datapoints)
			return [result], outlier
		return result

	def classify_column_proba(self, column, detect_outlier=False):
		nf = self.featuremap['number_feature']
		datapoints = nf.extract_features_column(column)
		result = self.clf.predict_proba(datapoints)
		prediction = self.clf.predict(datapoints)
		if detect_outlier:
			outlier_detector = self.outlier_detector_dict[prediction[0]]
			outlier = outlier_detector.predict_proba(datapoints)
			return result, outlier
		return result


	def classify_instance(self, entry):
		nf = self.featuremap['number_feature']
		datapoints = nf.extract_features_column([column])
		result = self.clf.predict(datapoints)
		return result

	def classify_instance_proba(self, entry):
		nf = self.featuremap['number_feature']
		datapoints = nf.extract_features_column([column])
		result = self.clf.predict_proba(datapoints)
		return result

	def classify_prepared_instance(self, entry):
		return self.clf.predict(entry.reshape(1, -1))[0]

	def execute_test(self, num_tests=5, learnset_ratio=0.7):
		number_feature = self.featuremap['number_feature']
		datapoints, targetpoints = number_feature.get_features_targets()
		return self.k_fold_test_classifier(self.create_datasets, [datapoints, targetpoints, learnset_ratio], \
			self.train_manual, self.classify_prepared_instance, num_tests)
