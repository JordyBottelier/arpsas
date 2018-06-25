from .feature_base import *

import string
import os
import sys
import numpy as np
import re
from random import shuffle

class Syntax_Feature_Model(Feature_Base):
	"""
		Feature class: Syntax Feature Model

		The Syntax Feature Model uses only single instances, thats why you should use num_columns = 1.
	"""


	def __init__(self, sf=None, num_columns=1, examples_per_column=0, unique=False, use_map=False):
		
		Feature_Base.__init__(self)

		if isinstance(sf, Storage_Files):
			self.dc = Data_Collector(sf, 
				num_columns=num_columns, examples_per_column=examples_per_column, unique=unique, use_map=use_map)
			self.columns = self.dc.get_columns()
		elif type(sf) == dict:
			self.columns=sf

		self.collect_features()

	def collect_features(self, use_self=True):
		""" For the entire dictionairy, collect the features for each column """

		corpus = []
		progress = 0
		for entity in self.columns:
			# There is only a single giant chunk for each entity so we take the first entry
			column = self.columns[entity][0]
			self.prep_columns(column)
			
			# Prep the features and their targets
			features = self.extract_features_column(column)
			self.features += features
			self.targets += [entity] * len(features)

			progress += (1 / float(len(self.columns))) * 100

	def extract_features_column(self, column):
		""" 
		From a column we extract the following syntactical features per entry:
			- Starts with a capital letter
			- multiple words
			- multiple capitalized words
			- Word is all uppercased
			- has special characters
			- has letters uppercase
			- has letters lowercase
			- has digits
			- entry is a digit


		This feature vector is a vector of booleans
		"""
		feature_column = []
		for entry in column:
			if entry != "": 
				feature_column.append(self.extract_features(entry))
		return feature_column

	def extract_features(self, entry):
		feature_vector = []
		feature_vector.append(entry[0].isupper())

		if len(entry.split(" ")) > 1:
			feature_vector.append(True)
			entry = entry.replace("  ", " ").strip()
			feature_vector.append(self.check_multiple_words_capitalized(entry.split(" ")))

		else:
			feature_vector.append(False)
			feature_vector.append(False)
		
		feature_vector.append(entry.isupper())
		feature_vector += self.check_special_characters(entry)
		feature_vector += self.check_upper_lower_digit(entry)
		feature_vector.append(entry.isdigit())
		feature_vector = np.array(feature_vector, dtype=int)
		return feature_vector

	def check_special_characters(self, entry):
		punctuation_list = []
		for char in string.punctuation:
			if char in entry:
				punctuation_list.append(True)
			else:
				punctuation_list.append(False)
		return punctuation_list

	def check_upper_lower_digit(self, word):
		has_upper = False
		has_lower = False
		has_digit = False

		for char in word:
			if char.isupper():
				has_upper = True
			if char.islower():
				has_lower = True
			if char.isdigit():
				has_digit = True
		return [has_upper, has_lower, has_digit]

	def check_multiple_words_capitalized(self, words):
		number_capitalized = 0
		words = list(filter(lambda a: a != "", words))
		for word in words:
			if word[0].isupper():
				number_capitalized += 1

		if number_capitalized > 1:
			return True
		else:
			return False




