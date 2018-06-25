
from .feature_base import *
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from random import shuffle
from sklearn.preprocessing import normalize, MinMaxScaler
import string
import numpy as np


class Number_Feature(Feature_Base):
	"""
		Number Feature: Used to heavily distinct between numbers (mostly kvk and SBI).

		Use the average number, character distribution of numbers, length of the integer/float, and if it is an 
		integer or float value (comma).

	"""

	def __init__(self, sf=None, num_columns=10, examples_per_column=0, unique=False, use_map=False):
		Feature_Base.__init__(self)
		if num_columns < 10:
			print("Warning, low amount of classes might cause the program to crash")

		if isinstance(sf, Storage_Files):
			self.dc = Data_Collector(sf, 
				num_columns=num_columns, examples_per_column=examples_per_column, unique=unique, use_map=use_map)
			self.columns = self.dc.get_columns()
		elif type(sf) == dict:
			self.columns=sf

		self.min_max_scaler = MinMaxScaler()
		self.collect_features()

	def collect_features(self):
		""" For the entire dictionairy, collect the features for each column """

		for entity in self.columns:
			column_chunks = self.columns[entity]
			for column in column_chunks:
				self.prep_columns(column)

				# Prep the features and their targets
				features = self.extract_features(column)
				self.features.append(features)
				self.targets.append(entity)

		self.features = self.min_max_scaler.fit_transform(self.features)

	def extract_features(self, column):
		""" 
		This function is used upon classification, we use this because we also need the fitted countvectorizor
		Extract raw features from column, three types:
			- char distributions
			- char-length metafeatures
			- token-length metafeatures
		"""
		
		feature_vector = []
		# feature_vector += self.char_distributions(column)
		# feature_vector += self.metafeatures(list(map(len, column))) # char length metafeatures
		num_col = self.get_number_column(column)
		feature_vector += self.metafeatures(num_col)
		feature_vector = np.array(feature_vector)
		return feature_vector

	def extract_features_column(self, column):
		""" 
		This function is used upon classification, we use this because we also need the fitted countvectorizor
		Extract raw features from column, three types:
			- char distributions
			- char-length metafeatures
			- token-length metafeatures
		"""
		
		feature_vector = []
		# feature_vector += self.char_distributions(column)
		# feature_vector += self.metafeatures(list(map(len, column))) # char length metafeatures
		num_col = self.get_number_column(column)
		feature_vector += self.metafeatures(num_col)
		feature_vector = np.array(feature_vector).reshape(1, -1)
		feature_vector = self.min_max_scaler.transform(feature_vector)
		return feature_vector

	def get_number_column(self, column):
		"""
			If it is possible, make all the string variables numerical so we can calculate the length and average number,
			highest and lowest, etc
		"""	
		result = []
		for entry in column:
			if isint(entry) and '.' not in entry:
				result.append(int(entry))
			elif isfloat(entry):
				result.append(float(entry))
			else:
				result.append(0)
		return result

	def metafeatures(self, a):
		""" Return metafeatures (i.e., descriptive statistics) of an array. """
		return [np.average(a), np.median(a), np.amin(a), np.amax(a)]

	def get_features_targets_test(self, learnset_ratio=0.7):
		"""
			Especially created for this fingerprint so the countvectorizer can also be tested along the
			rest of the datapoints.
		"""
		learnset_features = []
		learnset_targets = []

		testset_features = []
		testset_targets = []
		for entity in self.columns:
			column_chunks = self.columns[entity]
			for column in column_chunks:
				self.prep_columns(column)
				features_list = self.extract_features(column)
				# Split in learn and testset and prepare the features
				if np.random.uniform() < learnset_ratio:
					learnset_features.append(features_list)
					learnset_targets.append(entity)
				else:
					testset_features.append(features_list)
					testset_targets.append(entity)

		# Reinitialize minmaxscaler for testing
		self.min_max_scaler = MinMaxScaler()

		learnset_features = self.min_max_scaler.fit_transform(learnset_features)
		testset_features = self.min_max_scaler.transform(testset_features)

		return learnset_features, learnset_targets, testset_features, testset_targets

	def char_distributions(self, column):
		""" Counts over characters (i.e., distribution of chars per column). """
		c = Counter(string.punctuation + string.digits)
		c = dict.fromkeys(c, 0)

		for t in " ".join(column).lower():
			if t in c:
				c[t] += 1
		distribution = [val[1] for val in sorted(c.items())]
		distribution = np.array(distribution) / float(sum(distribution))
		return list(distribution)

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b
