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
	data_map_main = {
		'text': ['address', 'city', 'company_name', 'country', 'gender',\
			'legal_type', 'person_name', 'province', 'sbi_description'],
		'date_tel': ['date', 'telephone_nr'],
		'email': ['email'],
		'domain_name': ['domain_name'],
		'postcode': ['postcode'],
		'numbers': ['kvk_number', 'sbi_code', 'house_number']
	}
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
	sf_main = Storage_Files(data_folder, data_map_main)
	sf_numbers = Storage_Files(data_folder, ['kvk_number', 'sbi_code', 'house_number'])
	sf_text = Storage_Files(data_folder, ['address', 'city', 'company_name', 'country', 'gender',\
			'legal_type', 'person_name', 'province', 'sbi_description'])
	sf_date_tel = Storage_Files(data_folder, ['date', 'telephone_nr'])
	for i in range(0, rounds):
		ccc = Column_Classification_Config()
		# ------------------------------------------- CONFIG ------------------------------------------
		ccc.add_feature('main', 'Syntax_Feature_Model', [sf_main, 1, 5000, False, True])
		ccc.add_feature('numbers', 'Fingerprint', [sf_numbers, number_of_columns, 0, False, False])
		ccc.add_feature('name', 'Corpus', [sf_text, 100, 0, False, False])
		ccc.add_feature('date_tel', 'Fingerprint', [sf_date_tel, number_of_columns, examples_per_class, False, False])

		ccc.add_matcher('main', 'Syntax_Matcher', {'main': 'syntax'}) # main classifier
		ccc.add_matcher('numbers_matcher', 'Fingerprint_Matcher', {'numbers': 'fingerprint'}, ('main', 'numbers'))
		ccc.add_matcher('text_matcher', 'Word2Vec_Matcher', {'name': 'corpus'}, ('main', 'text'))
		ccc.add_matcher('date_tel_matcher', 'Fingerprint_Matcher', {'date_tel': 'fingerprint'}, ('main', 'date_tel'))
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
	data_map_main = {
		'text': ['address', 'city', 'company_name', 'country', 'gender',\
			'legal_type', 'person_name', 'province', 'sbi_description'],
		'date_tel': ['date', 'telephone_nr'],
		'email': ['email'],
		'domain_name': ['domain_name'],
		'postcode': ['postcode'],
		'numbers': ['kvk_number', 'sbi_code', 'house_number']
	}
	data_map_numbers = {
		'kvk_sbi': ['kvk_number', 'sbi_code'],
		'telephone_nr': ['telephone_nr'],
		'house_number': ['house_number']
	}
	data_folder = 'data_train/'

	number_of_columns = 80
	examples_per_class = 60
	total_actual = []
	total_predicted = []
	
	tmp = []
	exp_actual = []
	exp_predicted = []
	sf_main = Storage_Files(data_folder, data_map_main)
	sf_numbers = Storage_Files(data_folder, data_map_numbers)
	sf_text = Storage_Files(data_folder, ['address', 'city', 'company_name', 'country', 'gender',\
			'legal_type', 'person_name', 'province', 'sbi_description'])
	sf_date_tel = Storage_Files(data_folder, ['date', 'telephone_nr'])
	sf_kvk_sbi = Storage_Files(data_folder, ['kvk_number', 'sbi_code'])
	for i in range(0, rounds):
		ccc = Column_Classification_Config()
		# ------------------------------------------- CONFIG ------------------------------------------
		ccc.add_feature('main', 'Syntax_Feature_Model', [sf_main, 1, 5000, False, True])
		ccc.add_feature('numbers', 'Corpus', [sf_numbers, 100, 0, False, True])
		ccc.add_feature('name', 'Corpus', [sf_text, 100, 0, False, False])
		ccc.add_feature('date_tel', 'Fingerprint', [sf_date_tel, number_of_columns, examples_per_class, False, False])
		ccc.add_feature('kvk_sbi', 'Number_Feature', [sf_kvk_sbi, 100, 0, False, False])

		ccc.add_matcher('main', 'Syntax_Matcher', {'main': 'syntax'}) # main classifier
		ccc.add_matcher('numbers_matcher', 'Word2Vec_Matcher', {'numbers': 'corpus'}, ('main', 'numbers'))
		ccc.add_matcher('text_matcher', 'Word2Vec_Matcher', {'name': 'corpus'}, ('main', 'text'))
		ccc.add_matcher('date_tel_matcher', 'Fingerprint_Matcher', {'date_tel': 'fingerprint'}, ('main', 'date_tel'))
		ccc.add_matcher('kvk_sbi_matcher', 'Number_Matcher', {'kvk_sbi': 'number_feature'}, ('numbers_matcher', 'kvk_sbi'))
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

def experiment4_inliers3():
	data_map_main = {
		'text': ['address', 'city', 'company_name', 'country', 'gender',\
			'legal_type', 'person_name', 'province', 'sbi_description'],
		'date_tel': ['date', 'telephone_nr'],
		'email': ['email'],
		'domain_name': ['domain_name'],
		'postcode': ['postcode'],
		'numbers': ['kvk_number', 'sbi_code', 'house_number']
	}
	data_map_numbers = {
		'kvk_sbi': ['kvk_number', 'sbi_code'],
		'telephone_nr': ['telephone_nr'],
		'house_number': ['house_number']
	}

	data_map_text = {
		'place': ['city', 'province'],
		'country': ['country', 'country'],
		'gender': ['gender'],
		'legal_type': ['legal_type'],
		'pers_addr' : ['address', 'person_name'],
		'comp_addr': ['company_name', 'sbi_description']
	}
	data_folder = 'data_train/'

	number_of_columns = 80
	examples_per_class = 60
	total_actual = []
	total_predicted = []
	
	tmp = []
	exp_actual = []
	exp_predicted = []
	sf_main = Storage_Files(data_folder, data_map_main)
	sf_numbers = Storage_Files(data_folder, data_map_numbers)
	sf_text = Storage_Files(data_folder, data_map_text)
	sf_date_tel = Storage_Files(data_folder, ['date', 'telephone_nr'])
	sf_kvk_sbi = Storage_Files(data_folder, ['kvk_number', 'sbi_code'])
	sf_pers_addr = Storage_Files(data_folder, ['person_name', 'address', 'country'])

	sf_place = Storage_Files(data_folder, ['city', 'province', 'legal_type'])
	sf_comp_addr = Storage_Files(data_folder, ['company_name', 'sbi_description', 'country', 'legal_type'])
	for i in range(0, rounds):
		ccc = Column_Classification_Config()
		# ------------------------------------------- CONFIG ------------------------------------------
		ccc.add_feature('main', 'Syntax_Feature_Model', [sf_main, 1, 5000, False, True])
		# numbers
		ccc.add_feature('numbers', 'Corpus', [sf_numbers, 100, 0, False, True])
		ccc.add_feature('date_tel', 'Fingerprint', [sf_date_tel, number_of_columns, examples_per_class, False, False])
		ccc.add_feature('kvk_sbi', 'Number_Feature', [sf_kvk_sbi, 100, 0, False, False])

		# Text
		ccc.add_feature('name', 'Corpus', [sf_text, 100, 0, False, True])
		ccc.add_feature('feature_place', 'Fingerprint', [sf_place, number_of_columns, examples_per_class, False, False])
		ccc.add_feature('pers_addr', 'Fingerprint', [sf_pers_addr, number_of_columns, examples_per_class, False, False])
		ccc.add_feature('feature_comp_addr', 'Corpus', [sf_comp_addr, 100, 0, False, False])

		ccc.add_matcher('main', 'Syntax_Matcher', {'main': 'syntax'}) # main classifier
		# Numbers
		ccc.add_matcher('numbers_matcher', 'Word2Vec_Matcher', {'numbers': 'corpus'}, ('main', 'numbers'))
		ccc.add_matcher('date_tel_matcher', 'Fingerprint_Matcher', {'date_tel': 'fingerprint'}, ('main', 'date_tel'))
		ccc.add_matcher('kvk_sbi_matcher', 'Number_Matcher', {'kvk_sbi': 'number_feature'}, ('numbers_matcher', 'kvk_sbi'))

		# Text
		ccc.add_matcher('text_matcher', 'Word2Vec_Matcher', {'name': 'corpus'}, ('main', 'text'))
		ccc.add_matcher('match_place', 'Fingerprint_Matcher', {'feature_place': 'fingerprint'}, ('text_matcher', 'place'))
		ccc.add_matcher('match_pers_addr', 'Fingerprint_Matcher', {'pers_addr': 'fingerprint'}, ('text_matcher', 'pers_addr'))
		ccc.add_matcher('match_comp_addr', 'Word2Vec_Matcher', {'feature_comp_addr': 'corpus'}, ('text_matcher', 'comp_addr'))
		# ------------------------------------------- END CONFIG ------------------------------------------
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True, 0)
		# actual = get_confusion(predicted, actual, data_map_main)
		exp_actual += actual
		exp_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	gm.append_x(3)
	accuracy = round(sum(tmp) / float(rounds), 2)
	gm.append_y(accuracy)

	gm.store(filename="/graph_maker/exp1.4a_3")
	classnames = get_class_names(exp_actual)
	cm = confusion_matrix(exp_actual, exp_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True, title="Confusion Matrix Experiment 4a_3")
	subtitle = "Accuracy was averaged over " + str(rounds) + " tests"
	xticks_x = [0, 1, 2, 3]
	xticks_label = [0, 1, 2, 3]
	gm.plot_bar("Improvement Rounds", "Accuracy", "Accuracy of Pipeline over Rounds", subtitle=subtitle, show_value=True, \
		xticks_x=xticks_x, xticks_label=xticks_label, rotation=0)

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
	# gm = Graph_Maker()
	# gm.load(filename="/graph_maker/exp1.4a_1")
	# print(gm)
	# gm = Graph_Maker()
	# gm.load(filename="/graph_maker/exp1.4a_2")
	# print(gm)
	# gm = Graph_Maker()
	# gm.load(filename="/graph_maker/exp1.4a_3")
	# print(gm)