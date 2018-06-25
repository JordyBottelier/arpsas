"""
	This class is used to build the configuration for your match tree.
	It should be passed on to a schema matcher.  
"""
from .storage_files import Storage_Files
from .misc_func import *
from .feature_classes import *
from .column_classifiers import *
from .match_tree import *
from os.path import dirname, basename, isfile
import os
import glob
import re
import ast

import sys

class Column_Classification_Config():
	"""
		To build a matcher you first have to add feature objects. these can either collect data
		for you (using the data collector and storagefile object), or you can give the data in a dictionary. 
		You can list the possible features with self.print_feature_options.

		You can then add matchers, which only need a feature mapping. 
		
	"""

	def __init__(self):
		self.matcher_tree = Match_Tree()
		self.all_matcher_names = []
		self.feature_dict = {}
		self.possible_matchers = []
		self.possible_features = []
		self.feature_folder = 'feature_classes'
		self.matcher_folder = 'column_classifiers'
		self.unimport = ['Tester', 'Matcher', "Feature_Base", "Data_Collector"]
		workdir = glob.glob(dirname(__file__))[0].replace(os.getcwd(), '')[1::]
		self.build(workdir + '/' + self.matcher_folder, self.possible_matchers)
		self.build(workdir + '/' + self.feature_folder, self.possible_features)

	def build(self, folder, appendlist):
		"""
			Get all the possible classes from the files in the folder
		"""
		modules = glob.glob(folder+"/*.py")
		for f in modules:
			if isfile(f) and not f.endswith('__init__.py'):
				filename = basename(f)[:-3]
				texfile=open(f, "r")
				with open(f, 'r') as myfile:
					data = myfile.read()
					p = ast.parse(data)
					classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
					for n in classes:
						if n not in self.unimport:
							appendlist.append(n)

	def add_matcher(self, matcher_name, matcher_type, featuremap, parent_classes=('root', None)):
		"""
			A matcher uses the feature datapoints created by a feature class to train and predict. 
			A mapping is used in the matcher to get to the features has to be specified. 

			The featuremap should be a dictionary of which the keys are the names of the features you created
			using the add_feature method. The values should be the names used in the matchers (look them up).

			parent_classes should be a tuple with ('parent_matcher_name', 'class'), so the tree will know where
			to insert the matcher, and for which class. If it is a root matcher, use the tuple ('root', None) (or leave it empty).
			The matcher_name should be the matcher or submatcher of which this matcher will be the new submatcher. 
			The class parameter should be the class upon which this submatcher will be used. 
		"""
		tmp_dict = {}
		for key in featuremap:
			tmp_dict[featuremap[key]] = self.feature_dict[key]

		if matcher_type in self.possible_matchers:
			loc = {'featuremap': tmp_dict}
			exec("matcher = " + matcher_type + "(featuremap)", globals(), loc)
			matcher = loc['matcher']
			result = self.matcher_tree.add_matcher_to_tree(matcher_name, matcher, parent_classes)
		else:
			errorstring = "Matcher Class " + str(matcher_type) + " does not exist"
			throw_error(errorstring)


	def add_feature(self, feature_name, feature_type, arguments):
		"""
			Add a feature to the dictionary, data is loaded now. 
		"""
		if feature_name in self.feature_dict:
			print("Warning, double declaration of feature " + feature_name + ", overwriting it now")
		if feature_type in self.possible_features:
			loc = {'arguments': arguments}
			exec("feature = " + feature_type + "(*arguments)", globals(), loc)
			feature = loc['feature']
			self.feature_dict[feature_name] = feature
		else:
			errorstring = "Feature Class " + str(feature_type) + " does not exist"
			throw_error(errorstring)

	def get_tree(self):
		return self.matcher_tree

	def print_matcher_options(self):
		print(self.possible_matchers)

	def print_feature_options(self):
		print(self.possible_features)

	def __repr__ (self):
		print("Features:")
		print_dict(self.feature_dict)
		print("Matchers: ")
		self.matcher_tree.print_tree()
		print("Possible matchers/features:")
		print_dict(self.possible_matchers)
		print_dict(self.possible_features)
		return ""
