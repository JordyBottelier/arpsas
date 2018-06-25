from .feature_base import *

import string
import os
import sys
import numpy as np

class Corpus(Feature_Base):
	"""
		The corpus is a special kind of feature, it only removes the punctuation and lets you use the tokenized corpus
		of the column data as data for your matcher. 

		The features are stored as a list of tokens per defined entity.

		Instead of a storage file object you can also load the dictionary of columns per class directly. The format should be:
		{
			class1: [[col_data1], [col_data2]],
			class2: [[col_data3], [col_data4]]

		}
	"""

	def __init__(self, sf=None, num_columns=8, examples_per_column=0, unique=False, use_map=False):
		
		if isinstance(sf, Storage_Files):
			Feature_Base.__init__(self)
			self.dc = Data_Collector(sf, 
				num_columns=num_columns, examples_per_column=examples_per_column, unique=unique, use_map=use_map)
			self.columns = self.dc.get_columns()
		elif type(sf) == dict:
			self.columns=sf

		self.dim = 0
		self.collect_features()

	def collect_features(self):
		"""
			Process a column and add list of tokens to the corpus
		"""
		corpus = []
		targets = []
		for entity in self.columns:
			column_chunks = self.columns[entity]
			for column in column_chunks:
				self.prep_columns(column)
				corpus_column = self.extract_features_column(column)
				corpus.append(corpus_column)
				targets.append(entity)
		self.features = corpus
		self.targets = targets

	def extract_features_column(self, column):
		"""
			Replace punctuation with spaces and split on spaces. 
		"""
		result = []
		for entry in column:
			tmp = self.replace_punctuation(entry, ' ')
			for n in tmp.split():
				if n != "":
					result.append(n.lower())
		return result

	def replace_punctuation(self, entry, replacement=' '):
		"""
			Replace punctuation with other character
		"""
		exclude = set(string.punctuation)
		for char in exclude:
			entry = entry.replace(char, replacement)
		return entry

