"""
Performance experiment.

The code in all experiment files is very ugly, but it I did not see the use in creating beautifull
code for all experiments since it is a 'press enter and rerun them' setup.
"""

import time
from schema_matching import *
from schema_matching.misc_func import *
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import copy
from os.path import isfile
import os
import math
import sys
import numpy as np

def get_column(test_folder='data_test/'):
	sr = Schema_Reader() 
	col = []
	for filename in sorted(os.listdir(test_folder)):
		print(filename)
		path = test_folder + filename
		if(isfile(path)):
			headers, columns = sr.get_duplicate_columns(path, True)
			# print(columns[0])
			# print(headers[0])
			return columns[0]

def experiment5():
	data_folder = 'data_train/'
	class1 = ['city', 'country', 'date', 'gender', 'house_number',\
	'legal_type', 'postcode', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
	class2 = ['sbi_description']
	data_map = {
		"1": class1,
		"2": class2
	}
	gm_train_time = Graph_Maker()
	gm_classify_time = Graph_Maker()
	num_classifiers = 5
	matcher_classes = ["Fingerprint_Matcher", "Syntax_Matcher", "Word2Vec_Matcher"]
	feature_classes = ["Fingerprint", "Syntax_Feature_Model", "Corpus"]
	dict_args = ["fingerprint", "syntax", "corpus"]
	feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
	matcher_names = ['matcher1', 'matcher2', 'matcher3', 'matcher4', 'matcher5']
	sf_main = Storage_Files(data_folder, data_map)
	number_of_columns = [100, 1, 100]
	number_of_examples = [100, 10000, 100]
	col = get_column()
	
	for i in range(0, len(matcher_classes)):
		"""
			Loop through the matchers and execute the tests
		"""
		cccs = []
		train_times = []
		classification_times = []

		matcher_type = matcher_classes[i]
		print(matcher_type)
		feature_class = feature_classes[i]
		dict_arg = dict_args[i]
		ccc = Column_Classification_Config()
		num_cols = number_of_columns[i]
		num_ex = number_of_examples[i]
		total_time = 0
		for i in range(0, num_classifiers):
			"""
				Iteratively add a matcher to the block, we also add the matcher to a list to later test
				the speed. 
			"""
			feature_name = feature_names[i]
			matcher_name = matcher_names[i]
			elapsed = 0
			if i == 0:
				start = time.clock()
				ccc.add_feature(feature_name, feature_class, [sf_main, num_cols, num_ex, False, True])
				ccc.add_matcher(matcher_name, matcher_type, {feature_name: dict_arg}) # main classifier
				elapsed = time.clock() - start
			else:
				prev_matcher = matcher_names[i - 1]
				start = time.clock()
				ccc.add_feature(feature_name, feature_class, [sf_main, num_cols, num_ex, False, True])
				ccc.add_matcher(matcher_name, matcher_type, {feature_name: dict_arg}, (prev_matcher, '1')) # main classifier
				elapsed = time.clock() - start
			total_time += elapsed
			train_times.append(total_time)
			cccs.append(copy.deepcopy(ccc))

		
		for c in cccs:
			# print(c)
			# print("--------------")
			tmp = []
			for i in range(0, 100):
				sm = Schema_Matcher(c)
				start = time.clock()
				outcome = sm.classify_column([col])
				if outcome != "1":
					print("Not all matchers have been passed")
					sys.exit(0)
				elapsed = time.clock() - start
				tmp.append(elapsed)
			classification_times.append(np.mean(tmp))
		gm_classify_time.append_y(classification_times)
		gm_train_time.append_y(train_times)


	gm_train_time.add_x(list(range(1, num_classifiers + 1)))
	gm_classify_time.add_x(list(range(1, num_classifiers + 1)))
	print(gm_train_time)
	print(gm_classify_time)
	gm_train_time.store("/graph_maker/exp1.5_train")
	gm_classify_time.store("/graph_maker/exp1.5_class")
	
	xlabel = "Number of Matchers"
	ylabel = "Time in Seconds"
	subtitle = "Number of Colums: " + str(number_of_columns[0]) + "\nNumber of Examples per Column: " + str(number_of_examples[0])
	gm_train_time.plot_line_n(xlabel, "Training " + ylabel, "Number of Matchers vs Training Time", feature_classes ,subtitle=subtitle)
	gm_classify_time.plot_line_n(xlabel, "Classification " + ylabel, "Number of Matchers vs Classification Time", \
		feature_classes ,subtitle=subtitle)

if __name__ == '__main__':
	experiment5()
	# get_column()