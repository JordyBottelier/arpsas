"""
Pipelines experiment.

The code in all experiment files is very ugly, but it I did not see the use in creating beautifull
code for all experiments since it is a 'press enter and rerun them' setup.
"""

from schema_matching import *
from schema_matching.misc_func import *
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from os.path import isfile
import os
import math


gm = Graph_Maker()
rounds = 3

def execute_test(sm, test_folder, skip_unknown=False, iterations=0):
	"""
		for all the schemas in the test folder, read them and classify them, 
		also compute precision, recall, f_measure and accuracy. 
	"""
	sr = Schema_Reader()
	actual = []
	predicted = []
	i = 0
	for filename in sorted(os.listdir(test_folder)):
		i += 1
		print(filename)
		path = test_folder + filename
		if(isfile(path)):
			headers, columns = sr.get_duplicate_columns(path, skip_unknown)
			result_headers = None
			if skip_unknown:
				result_headers = sm.test_schema_matcher(columns, 0, False)
			else:
				result_headers = sm.test_schema_matcher(columns, 0.4, True)
			predicted += result_headers
			actual += headers
			# print(accuracy_score(actual, predicted))
		# break
		if i == iterations:
			break
	return actual, predicted

def get_confusion(pred, actual , data_map):
	"""
		Modify the actual classes according to the datamap, so we can look at the confusion matrix.
	"""
	result = []
	for ac in actual:
		for cl in data_map:
			if ac in data_map[cl]:
				result.append(cl)
	for i in range(0, len(actual)):
		if pred[i] != result[i]:
			print(actual[i])
	return result



def experiment4_inliers1():
	data_folder = 'data_train/'

	number_of_columns = 80
	examples_per_class = 60
	gm.append_x(0)
	gm.append_y(0.88)
	total_actual = []
	total_predicted = []
	
	tmp = []
	exp_actual = []
	exp_predicted = []
	sf_main = Storage_Files(data_folder, ['city', 'country', 'date', 'gender', 'house_number',\
		'legal_type', 'province', 'sbi_code', 'sbi_description', 'telephone_nr', 'postcode'])
	sf_legal = Storage_Files(data_folder, ['legal_type', 'postcode'])
	sf_province = Storage_Files(data_folder, ['province', 'postcode'])
	for i in range(0, rounds):
		ccc = Column_Classification_Config()
		# ------------------------------------------- CONFIG ------------------------------------------
		ccc.add_feature('main', 'Corpus', [sf_main, 60, 0, False, False])
		ccc.add_feature('legal', 'Syntax_Feature_Model', [sf_legal, 1, 0, False, False])
		ccc.add_feature('province', 'Syntax_Feature_Model', [sf_province, 1, 0, False, False])

		ccc.add_matcher('main', 'Word2Vec_Matcher', {'main': 'corpus'}) # main classifier
		ccc.add_matcher('legal_matcher', 'Syntax_Matcher', {'legal': 'syntax'}, ('main', 'legal_type'))
		ccc.add_matcher('province_matcher', 'Syntax_Matcher', {'province': 'syntax'}, ('main', 'province'))
		# ccc.add_matcher('dom_email_matcher', 'Syntax_Matcher', {'dom_email': 'syntax'}, ('main', 'domain_email'))
		# ------------------------------------------- END CONFIG ------------------------------------------
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True, 0)
		# actual = get_confusion(predicted, actual, data_map_main)
		exp_actual += actual
		exp_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	gm.append_x(1)
	accuracy = round(sum(tmp) / float(rounds), 2)
	gm.append_y(accuracy)

	gm.store(filename="/graph_maker/exp1.4a_1")
	classnames = get_class_names(exp_actual)
	cm = confusion_matrix(exp_actual, exp_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True, title="Confusion Matrix Experiment 4a_1")
	subtitle = "Accuracy was averaged over " + str(rounds) + " tests"

def experiment4_inliers2():
	data_folder = 'data_train/'

	number_of_columns = 80
	examples_per_class = 60
	gm.append_x(0)
	gm.append_y(0.88)
	total_actual = []
	total_predicted = []
	
	tmp = []
	exp_actual = []
	exp_predicted = []
	classes = ['city', 'country', 'date', 'gender', 'house_number',\
		'legal_type', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
	sf_main = Storage_Files(data_folder, ['city', 'country', 'date', 'gender', 'house_number',\
		'legal_type', 'province', 'sbi_code', 'sbi_description', 'telephone_nr', 'postcode'])
	sf_all = Storage_Files(data_folder, classes)
	for i in range(0, rounds):
		ccc = Column_Classification_Config()
		# ------------------------------------------- CONFIG ------------------------------------------
		ccc.add_feature('main', 'Syntax_Feature_Model', [sf_main, 1, 5000, False, False])
		ccc.add_feature('all', 'Corpus', [sf_all, 50, 0, False, False])
		ccc.add_feature('city', 'Corpus', [sf_city, 50, 0, False, False])

		ccc.add_matcher('main', 'Syntax_Matcher', {'main': 'syntax'}) # main classifier
		ccc.add_matcher('legal_matcher', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'legal_type'))
		ccc.add_matcher('1', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'city'))
		ccc.add_matcher('2', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'country'))
		ccc.add_matcher('3', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'date'))
		ccc.add_matcher('4', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'gender'))
		ccc.add_matcher('5', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'house_number'))
		ccc.add_matcher('6', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'province'))
		ccc.add_matcher('7', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'sbi_code'))
		ccc.add_matcher('8', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'sbi_description'))
		ccc.add_matcher('9', 'Word2Vec_Matcher', {'all': 'corpus'}, ('main', 'telephone_nr'))

		# ------------------------------------------- END CONFIG ------------------------------------------
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True, 0)
		# actual = get_confusion(predicted, actual, data_map_main)
		exp_actual += actual
		exp_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	gm.append_x(2)
	accuracy = round(sum(tmp) / float(rounds), 2)
	gm.append_y(accuracy)

	gm.store(filename="/graph_maker/exp1.4a_2")
	classnames = get_class_names(exp_actual)
	cm = confusion_matrix(exp_actual, exp_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True, title="Confusion Matrix Experiment 4a_2")
	subtitle = "Accuracy was averaged over " + str(rounds) + " rounds"

def get_class_names(ytrue):
	res = []
	for c in ytrue:
		if c not in res:
			res.append(c)
	return res

if __name__ == '__main__':
	experiment4_inliers1()
	experiment4_inliers2()
	experiment4_inliers3()