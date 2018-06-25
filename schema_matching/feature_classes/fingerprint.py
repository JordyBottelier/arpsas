from .feature_base import *

import string
import os
import sys
import numpy as np
import re
from collections import Counter
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from random import shuffle
from sklearn.preprocessing import normalize
import copy

from operator import methodcaller

class Fingerprint(Feature_Base):
	"""
		Feature class: Fingerprint

		Uses the distribution of characters and an n gram of 2 letters in its corpus as 
		datapoints. It also uses metadata of these distributions
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

		self.cv = CountVectorizer(analyzer = 'char_wb', 
                               ngram_range = (1, 2), 
                               min_df = 5, 
                               decode_error = 'ignore')
		self.collect_features()

	def collect_features(self):
		""" For the entire dictionairy, collect the features for each column """
		corpus = []
		for entity in self.columns:
			column_chunks = self.columns[entity]
			for column in column_chunks:
				self.prep_columns(column)
				corpus_txt = " ".join(column)
				corpus.append(corpus_txt)
				
				# Prep the features and their targets
				features = self.extract_features(column)
				self.features.append(features)
				self.targets.append(entity)

		n_grams = self.cv.fit_transform(corpus).toarray()
		self.features = np.hstack((self.features, n_grams))
		self.features = normalize(self.features)

	def get_features_targets_test(self, learnset_ratio=0.7):
		"""
			Especially created for this fingerprint so the countvectorizer can also be tested along the
			rest of the datapoints.
		"""
		learnset_corpus = []
		learnset_features = []
		learnset_targets = []

		testset_corpus = []
		testset_features = []
		testset_targets = []
		for entity in self.columns:
			column_chunks = self.columns[entity]
			for column in column_chunks:
				self.prep_columns(column)
				features_list = self.extract_features(column)
				corpus_txt = " ".join(column)
				# Split in learn and testset and prepare the features
				if np.random.uniform() < learnset_ratio:
					learnset_corpus.append(corpus_txt)
					learnset_features.append(features_list)
					learnset_targets.append(entity)
				else:
					testset_corpus.append(corpus_txt)
					testset_features.append(features_list)
					testset_targets.append(entity)

		# Reinitialize countvectorizer for testing
		self.cv = CountVectorizer(analyzer = 'char_wb', 
                               ngram_range = (1, 2), 
                               min_df = 5, 
                               decode_error = 'ignore')

		learn_n_grams = self.cv.fit_transform(learnset_corpus).toarray()
		learnset_features = np.hstack((learnset_features, learn_n_grams))
		learnset_features = normalize(learnset_features)
		
		test_n_grams = self.cv.transform(testset_corpus).toarray()
		testset_features = np.hstack((testset_features, test_n_grams))
		testset_features = normalize(testset_features)
		return learnset_features, learnset_targets, testset_features, testset_targets


	def extract_features(self, column):
		""" 
		Extract raw features from column, three types:
			- char distributions
			- char-length metafeatures
			- token-length metafeatures
		"""

		feature_vector = []
		feature_vector += self.char_distributions(column)
		feature_vector += self.metafeatures(list(map(len, column))) # char length metafeatures
		words_per_entry = list(map(len, list(map(methodcaller("split", " "), column)))) # Get the amount of words per entry
		feature_vector += self.metafeatures(words_per_entry) # Amount of words per entry metafeatures
		return np.array(feature_vector)

	def extract_features_column(self, column):
		""" 
		This function is used upon classification, we use this because we also need the fitted countvectorizor
		Extract raw features from column, three types:
			- char distributions
			- char-length metafeatures
			- token-length metafeatures
		"""
		
		feature_vector = []
		feature_vector += self.char_distributions(column)
		feature_vector += self.metafeatures(list(map(len, column))) # char length metafeatures
		words_per_entry = list(map(len, list(map(methodcaller("split", " "), column)))) # Get the amount of words per entry
		feature_vector += self.metafeatures(words_per_entry) # Amount of words per entry metafeatures
		ngrams = self.cv.transform([" ".join(column)]).toarray()[0].tolist()
		feature_vector += ngrams
		feature_vector = np.array(feature_vector).reshape(1, -1)
		return np.array(normalize(feature_vector))

	def metafeatures(self, a):
		""" Return metafeatures (i.e., descriptive statistics) of an array. """
		return [np.average(a), np.median(a), np.std(a), np.amin(a), 
				np.amax(a), np.percentile(a, 25), np.percentile(a, 75)]

	def char_distributions(self, column):
		""" Counts over characters (i.e., distribution of chars per column). """
		c = Counter(string.punctuation + string.ascii_letters.lower() + string.digits)
		c = dict.fromkeys(c, 0)

		for t in " ".join(column).lower():
			if t in c:
				c[t] += 1

		items = sorted(c.items(), key=lambda tup: tup[0])
		distribution = [val[1] for val in items]
		distribution = np.array(distribution) / float(sum(distribution))
		return list(distribution)





