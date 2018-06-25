

from .match_base import *
import numpy as np
from .tester import Tester



class Fingerprint_Matcher(Matcher, Tester):

	def __init__(self, featuremap):
		Matcher.__init__(self, featuremap)
		self.train()

	def train(self):
		fingerprint = self.featuremap['fingerprint']
		datapoints, targetpoints = fingerprint.get_features_targets()
		self.create_oneclass_dict(datapoints, targetpoints)
		self.train_manual(datapoints, targetpoints)

	def train_manual(self, datapoints, targetpoints):
		self.clf.fit(datapoints, targetpoints)

	def classify_column(self, column, detect_outlier=False):
		""" The column should be a list with strings to classify """
		fingerprint = self.featuremap['fingerprint']
		datapoints = fingerprint.extract_features_column(column)
		prediction = self.clf.predict(datapoints)
		if detect_outlier:
			outlier = self.outlier_detector_dict[prediction[0]].predict(datapoints)
			return self.clf.predict(datapoints), outlier
		return prediction

	def classify_instance(self, entry):
		""" Classify a single instance """
		fingerprint = self.featuremap['fingerprint']
		datapoints = fingerprint.extract_features_column([entry])
		return self.clf.predict(datapoints)

	def classify_prepared_instance(self, entry):
		return self.clf.predict(entry.reshape(1, -1))[0]

	def classify_column_proba(self, column, detect_outlier=False):
		""" Return the classification probability model for a column """
		fingerprint = self.featuremap['fingerprint']
		datapoints = fingerprint.extract_features_column(column)
		prediction = self.clf.predict(datapoints)
		if detect_outlier:
			outlier = self.outlier_detector_dict[prediction[0]].predict_proba(datapoints)
			return self.clf.predict_proba(datapoints), outlier
		return self.clf.predict_proba(datapoints)

	def classify_instance_proba(self, entry):
		""" Return the classification probability model for an instance """
		fingerprint = self.featuremap['fingerprint']
		datapoints = fingerprint.extract_features_column([entry])
		return self.clf.predict_proba(datapoints)

	def execute_test(self, num_tests=5, learnset_ratio=0.7):
		fingerprint = self.featuremap['fingerprint']
		return self.k_fold_test_classifier(fingerprint.get_features_targets_test, [learnset_ratio], 
			self.train_manual, self.classify_prepared_instance, num_tests)

	def execute_test_incremental(self, num_tests=5, learnset_ratio=0.7):
		fingerprint = self.featuremap['fingerprint']
		return self.k_fold_test_incremental_random(fingerprint.get_features_targets_test, [learnset_ratio], 
			self.train_manual, self.classify_prepared_instance, num_tests)


		
