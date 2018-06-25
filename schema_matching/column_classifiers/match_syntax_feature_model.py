from .match_base import *
import numpy as np
from .tester import Tester

class Syntax_Matcher(Matcher, Tester):

	def __init__(self, featuremap):
		Matcher.__init__(self, featuremap)
		self.train()

	def train(self):
		syntax_feature = self.featuremap['syntax']
		datapoints, targetpoints = syntax_feature.get_features_targets()
		svm = LinearSVC(class_weight = 'balanced', multi_class='crammer_singer')
		self.clf = CalibratedClassifierCV(svm)
		self.train_manual(datapoints, targetpoints)
		self.create_oneclass_dict(datapoints, targetpoints)

	def train_manual(self, datapoints, targetpoints):
		self.clf.fit(datapoints, targetpoints)

	def classify_column(self, column, detect_outlier=False):
		""" 
			The column should be a list with strings to classify.
			Here we can create our own custom algorithm to match a column. 
		"""
		syntax = self.featuremap['syntax']
		datapoints = syntax.extract_features_column(column)
		result = self.clf.predict(datapoints)
		result = self.most_common(result)
		if detect_outlier:
			outlier = self.outlier_detector_dict[result].predict(datapoints)
			outlier = [np.mean(outlier, axis=0)]
			return [result], outlier
		return result

	def classify_column_proba(self, column, detect_outlier=False):
		"""
			Return the mean classification probability, and return it in the standard
			numpy format [[]]
		"""
		syntax = self.featuremap['syntax']
		datapoints = syntax.extract_features_column(column)
		result = self.clf.predict(datapoints)
		probabilities = self.clf.predict_proba(datapoints)
		result = self.most_common(result)
		prediction = self.clf.predict(datapoints)
		probs = np.mean(probabilities, axis=0)
		if detect_outlier:
			outlier = self.outlier_detector_dict[result].predict_proba(datapoints)
			outlier = [np.mean(outlier, axis=0)]
			return [probs], outlier
		return [probs]

	def classify_instance_proba(self, entry):
		""" Give prediction probability of single instance classification """
		syntax = self.featuremap['syntax']
		datapoints = syntax.extract_features_column([entry])
		return self.clf.predict_proba(datapoints)

	def classify_instance(self, entry):
		""" Classify a single instance """
		syntax = self.featuremap['syntax']
		datapoints = syntax.extract_features_column([entry])
		return self.clf.predict(datapoints)

	def classify_prepared_instance(self, entry):
		return self.clf.predict(entry.reshape(1, -1))[0]

	def execute_test(self, num_tests=5, learnset_ratio=0.7):
		syntax_feature = self.featuremap['syntax']
		datapoints, targetpoints = syntax_feature.get_features_targets()
		return self.k_fold_test_classifier(self.create_datasets, [datapoints, targetpoints, learnset_ratio], \
			self.train_manual, self.classify_prepared_instance, num_tests)

	def execute_test_incremental(self, num_tests=5, learnset_ratio=0.7):
		syntax_feature = self.featuremap['syntax']
		datapoints, targetpoints = syntax_feature.get_features_targets()
		return self.k_fold_test_incremental_random(self.create_datasets, [datapoints, targetpoints, learnset_ratio], \
			self.train_manual, self.classify_prepared_instance, num_tests)
