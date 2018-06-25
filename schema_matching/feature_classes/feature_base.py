from .data_collector import Data_Collector
from ..misc_func import *
from abc import ABCMeta, abstractmethod
from ..storage_files import Storage_Files


class Feature_Base(metaclass=ABCMeta):
	""" 
		Every feature should create its own datapoints upon initiation and store them in variables called features
		and targets. The data for the features are gathered by the data collector, which uses a storage file object to gather
		data. 
	"""

	def __init__(self):
		
		self.features = []
		self.targets = []

	def prep_columns(self, column):
		""" Make sure every element of a column is a string.
		"""
		for i in range(0, len(column)):
			column[i] = str(column[i])

	def get_features_targets(self):
		return self.features, self.targets

	@abstractmethod
	def extract_features_column(self, column):
		""" Needed in the matchers """
		pass
