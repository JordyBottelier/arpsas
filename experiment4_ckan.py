"""
Pipelines experiment for ckan data.

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
import string

headers_dict = {   'has_description': 2400,
	'has_identifier_has_URI': 2400,
	'has_identifier_has_id_value': 2400,
	'has_identifier_is_source_of_has_classification_type': 804,
	'has_identifier_is_source_of_has_endDate': 804,
	'has_identifier_is_source_of_has_startDate': 804,
	'has_identifier_is_source_of_type': 804,
	'has_identifier_label': 2400,
	'has_identifier_type': 2400,
	'has_name': 2400,
	'is_destination_of_has_classification_type': 2400,
	'is_destination_of_has_endDate': 988,
	'is_destination_of_has_source_has_identifier_has_URI': 924,
	'is_destination_of_has_source_has_identifier_has_id_value': 183,
	'is_destination_of_has_source_has_identifier_type': 924,
	'is_destination_of_has_source_is_source_of_has_classification_type': 952,
	'is_destination_of_has_source_is_source_of_has_destination_has_name': 141,
	'is_destination_of_has_source_is_source_of_has_destination_type': 952,
	'is_destination_of_has_source_is_source_of_has_endDate': 952,
	'is_destination_of_has_source_is_source_of_has_startDate': 952,
	'is_destination_of_has_source_is_source_of_type': 952,
	'is_destination_of_has_source_type': 2400,
	'is_destination_of_has_startDate': 988,
	'is_destination_of_type': 2400,
	'is_source_of_has_classification_type': 2400,
	'is_source_of_has_destination_type': 1679,
	'is_source_of_has_endDate': 2400,
	'is_source_of_has_startDate': 2400,
	'is_source_of_type': 2400,
	'label': 2400,
	'type': 2400}

data_map_main = {
	'has_description': ['has_description'],
	'has_identifier_has_URI': ['has_identifier_has_URI'],
	'has_identifier_has_id_value': ['has_identifier_has_id_value'],
	'has_identifier_is_source_of_has_classification_type': ['has_identifier_is_source_of_has_classification_type'],
	'has_identifier_is_source_of_has_endDate': ['has_identifier_is_source_of_has_endDate'],
	'has_identifier_is_source_of_has_startDate': ['has_identifier_is_source_of_has_startDate'],
	'has_identifier_is_source_of_type': ['has_identifier_is_source_of_type'],
	'has_identifier_label': ['has_identifier_label'],
	'has_identifier_type': ['has_identifier_type'],
	'has_name': ['has_name'],
	'is_destination_of_has_classification_type': ['is_destination_of_has_classification_type'],
	'is_destination_of_has_endDate': ['is_destination_of_has_endDate'],
	'is_destination_of_has_source_has_identifier_has_URI': ['is_destination_of_has_source_has_identifier_has_URI'],
	'is_destination_of_has_source_has_identifier_has_id_value': ['is_destination_of_has_source_has_identifier_has_id_value'],
	'is_destination_of_has_source_has_identifier_type': ['is_destination_of_has_source_has_identifier_type'],
	'is_destination_of_has_source_is_source_of_has_classification_type':\
	['is_destination_of_has_source_is_source_of_has_classification_type'],
	'is_destination_of_has_source_is_source_of_has_destination_has_name': \
	['is_destination_of_has_source_is_source_of_has_destination_has_name'],
	'is_destination_of_has_source_is_source_of_has_destination_type': \
	['is_destination_of_has_source_is_source_of_has_destination_type'],
	'is_destination_of_has_source_is_source_of_has_endDate': [ 'is_destination_of_has_source_is_source_of_has_endDate'],
	'is_destination_of_has_source_is_source_of_has_startDate': ['is_destination_of_has_source_is_source_of_has_startDate'],
	'is_destination_of_has_source_is_source_of_type': ['is_destination_of_has_source_is_source_of_type'],
	'is_destination_of_has_source_type': ['is_destination_of_has_source_type'],
	'is_destination_of_has_startDate': ['is_destination_of_has_startDate'],
	'is_destination_of_type': ['is_destination_of_type'],
	'is_source_of_has_classification_type': ['is_source_of_has_classification_type'],
	'is_source_of_has_destination_type': ['is_source_of_has_destination_type'],
	'is_source_of_has_endDate': ['is_source_of_has_endDate'],
	'is_source_of_has_startDate': ['is_source_of_has_startDate'],
	'is_source_of_type': ['is_source_of_type'],
	'label': ['label'],
	'type': ['type']
	}


classes = os.listdir("ckan_subset/prepared_learnset/")
gm = Graph_Maker()
rounds = 3


def get_letter(classname):
	mapping = read_file("ckan_subset/classname_reverse")
	letter_to_ckan = read_file("ckan_subset/classname_map")
	# print_dict(mapping)
	try:
		return mapping[classname]
	except: # class is apperently not present
		alphabet = list(string.ascii_uppercase + string.ascii_lowercase)
		for letter in alphabet:
			if letter not in letter_to_ckan:
				mapping[classname] = letter
				letter_to_ckan[letter] = classname
				store_file("ckan_subset/classname_map", letter_to_ckan)
				store_file("ckan_subset/classname_reverse", mapping)

				return mapping[classname]

def get_class_names(ytrue):
	res = []
	for c in ytrue:
		if c not in res:
			res.append(c)
	return res

def execute_test_ckan(sm, test_folder, skip_unknown=False):
	"""
		for all the schemas in the test folder, read them and classify them, 
		also compute precision, recall, f_measure and accuracy. 
	"""
	sr = Schema_Reader()
	actual = []
	predicted = []
	for filename in sorted(os.listdir(test_folder)):
		print(filename)
		path = test_folder + filename
		# try:
		if(isfile(path)):
			headers, columns = sr.get_duplicate_columns(path, skip_unknown)
			result_headers = None
			if skip_unknown:
				# we have to do this since there are non-unknown headers in the testset which are not present in the learnset
				tmp = [] 
				tmp2 = []
				for i in range(0, len(headers)):
					header = headers[i]
					column = columns[i]
					if header in headers_dict:
						tmp.append(header)
						tmp2.append(column)
				headers = tmp
				columns = tmp2
				result_headers = sm.test_schema_matcher(columns, 0, False)
			else:
				for i in range(0, len(headers)):
					header = headers[i]
					if header not in headers_dict:
						headers[i] = 'unknown'
				result_headers = sm.test_schema_matcher(columns, 0.5, True)
			for head in result_headers:
				predicted.append(get_letter(head))
			for head in headers:
				actual.append(get_letter(head))
		# except:
		# 	print("Fail")
		# break
	# print(ConfusionMatrix(actual, predicted))
	return actual, predicted

def experiment4_inliers():
	data_folder = "ckan_subset/prepared_learnset/"
	test_folder = 'ckan_subset/testset/xml_csv/'
	gm.append_x(0)

	xticks_x = [0, 1]
	xticks_label = ["Baseline", "Pipeline"]
	data_map_first = {
	'has_identifier_has_URI': ['has_identifier_has_URI'],
	'has_identifier_is_source_of_has_classification_type': ['has_identifier_is_source_of_has_classification_type'],
	'has_identifier_is_source_of_has_endDate': ['has_identifier_is_source_of_has_endDate'],
	'has_identifier_is_source_of_has_startDate': ['has_identifier_is_source_of_has_startDate'],
	'has_identifier_is_source_of_type': ['has_identifier_is_source_of_type'],
	'has_identifier_label': ['has_identifier_label'],
	'has_identifier_type': ['has_identifier_type'],
	'is_destination_of_has_classification_type': ['is_destination_of_has_classification_type'],
	'is_destination_of_has_endDate': ['is_destination_of_has_endDate'],
	'is_destination_of_has_source_has_identifier_has_URI': ['is_destination_of_has_source_has_identifier_has_URI'],
	'is_destination_of_has_source_has_identifier_has_id_value': ['is_destination_of_has_source_has_identifier_has_id_value'],
	'is_destination_of_has_source_has_identifier_type': ['is_destination_of_has_source_has_identifier_type'],
	'is_destination_of_has_source_is_source_of_has_classification_type':\
	['is_destination_of_has_source_is_source_of_has_classification_type'],
	'is_destination_of_has_source_is_source_of_has_destination_type': \
	['is_destination_of_has_source_is_source_of_has_destination_type'],
	'is_destination_of_has_source_is_source_of_has_endDate': [ 'is_destination_of_has_source_is_source_of_has_endDate'],
	'is_destination_of_has_source_is_source_of_has_startDate': ['is_destination_of_has_source_is_source_of_has_startDate'],
	'is_destination_of_has_source_is_source_of_type': ['is_destination_of_has_source_is_source_of_type'],
	'is_destination_of_has_source_type': ['is_destination_of_has_source_type'],
	'is_destination_of_has_startDate': ['is_destination_of_has_startDate'],
	'is_destination_of_type': ['is_destination_of_type'],
	'is_source_of_has_classification_type': ['is_source_of_has_classification_type'],
	'is_source_of_has_destination_type': ['is_source_of_has_destination_type'],
	'is_source_of_has_endDate': ['is_source_of_has_endDate'],
	'is_source_of_has_startDate': ['is_source_of_has_startDate'],
	'is_source_of_type': ['is_source_of_type'],
	'label': ['label'],
	'type': ['type'],
	'send_second': ['is_destination_of_has_source_is_source_of_has_destination_has_name', 'has_identifier_has_id_value',\
	'has_description', 'has_name'] 
	}
	data_map_sfm = {
		'has_description': ['has_description'],
		'is_destination_of_has_source_is_source_of_has_destination_has_name': \
		['is_destination_of_has_source_is_source_of_has_destination_has_name'],
		'send_third': ['has_name', 'has_identifier_has_id_value']
	}

	number_of_classes = 15
	examples_per_class = 0

	accuracies = []
	total_actual = []
	total_predicted = []
	# accuracies = [0.4, 0.4, 0.4]
	sf_main = Storage_Files(data_folder, data_map_first)
	sf_sfm = Storage_Files(data_folder, data_map_sfm)
	sf_third = Storage_Files(data_folder, ['has_name', 'has_identifier_has_id_value'])
	tmp = []
	for i in range(0, rounds):
		print("Fingerprint")
		# --- Fingerprint
		ccc = Column_Classification_Config()
		# Features
		ccc.add_feature('feature_main', 'Fingerprint', [sf_main, 15, 0, False, True])
		ccc.add_feature('feature_second', 'Corpus', [sf_sfm, 15, 0, False, True])
		ccc.add_feature('feature_third', 'Syntax_Feature_Model', [sf_third, 1, 0, False, False])
		

		# Matchers
		ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
		ccc.add_matcher('second', 'Word2Vec_Matcher', {'feature_second': 'corpus'}, ('matcher', 'send_second'))
		ccc.add_matcher('third', 'Syntax_Matcher', {'feature_third': 'syntax'}, ('second', 'send_third'))
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test_ckan(sm, test_folder, True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	gm.append_x(1)
	gm.append_y(0.76)
	accuracy = round(sum(tmp) / float(rounds), 2)
	gm.append_y(accuracy)
	
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_small_font(cm, classnames, normalize=True, title="Confusion Matrix Experiment 4 CKAN Dataset")

	gm.store(filename="/graph_maker/exp1.4ckan_a_1")
	subtitle = "Accuracy was averaged over " + str(rounds) + " tests"
	gm.plot_bar("Improvement Rounds", "Accuracy", "Accuracy of Pipeline over Rounds", subtitle=subtitle, show_value=True, \
		xticks_x=xticks_x, xticks_label=xticks_label, rotation=0)
	labels = ["Accuracy", "Precision", "Recall", "F-Measure"]


def experiment4_outliers():
	data_folder = "ckan_subset/prepared_learnset/"
	test_folder = 'ckan_subset/testset/xml_csv/'
	accuracies = []
	precisions = []
	recalls = []
	f_measures = []

	accuracies.append(0.83)
	precisions.append(0.54)
	recalls.append(0.35)
	f_measures.append(0.42)
	gm.append_x("Baseline")
		
	data_map_first = {
	'has_identifier_has_URI': ['has_identifier_has_URI'],
	'has_identifier_is_source_of_has_classification_type': ['has_identifier_is_source_of_has_classification_type'],
	'has_identifier_is_source_of_has_endDate': ['has_identifier_is_source_of_has_endDate'],
	'has_identifier_is_source_of_has_startDate': ['has_identifier_is_source_of_has_startDate'],
	'has_identifier_is_source_of_type': ['has_identifier_is_source_of_type'],
	'has_identifier_label': ['has_identifier_label'],
	'has_identifier_type': ['has_identifier_type'],
	'is_destination_of_has_classification_type': ['is_destination_of_has_classification_type'],
	'is_destination_of_has_endDate': ['is_destination_of_has_endDate'],
	'is_destination_of_has_source_has_identifier_has_URI': ['is_destination_of_has_source_has_identifier_has_URI'],
	'is_destination_of_has_source_has_identifier_has_id_value': ['is_destination_of_has_source_has_identifier_has_id_value'],
	'is_destination_of_has_source_has_identifier_type': ['is_destination_of_has_source_has_identifier_type'],
	'is_destination_of_has_source_is_source_of_has_classification_type':\
	['is_destination_of_has_source_is_source_of_has_classification_type'],
	'is_destination_of_has_source_is_source_of_has_destination_type': \
	['is_destination_of_has_source_is_source_of_has_destination_type'],
	'is_destination_of_has_source_is_source_of_has_endDate': [ 'is_destination_of_has_source_is_source_of_has_endDate'],
	'is_destination_of_has_source_is_source_of_has_startDate': ['is_destination_of_has_source_is_source_of_has_startDate'],
	'is_destination_of_has_source_is_source_of_type': ['is_destination_of_has_source_is_source_of_type'],
	'is_destination_of_has_source_type': ['is_destination_of_has_source_type'],
	'is_destination_of_has_startDate': ['is_destination_of_has_startDate'],
	'is_destination_of_type': ['is_destination_of_type'],
	'is_source_of_has_classification_type': ['is_source_of_has_classification_type'],
	'is_source_of_has_destination_type': ['is_source_of_has_destination_type'],
	'is_source_of_has_endDate': ['is_source_of_has_endDate'],
	'is_source_of_has_startDate': ['is_source_of_has_startDate'],
	'is_source_of_type': ['is_source_of_type'],
	'label': ['label'],
	'type': ['type'],
	'send_second': ['is_destination_of_has_source_is_source_of_has_destination_has_name', 'has_identifier_has_id_value',\
	'has_description', 'has_name'] 
	}
	data_map_sfm = {
		'has_description': ['has_description'],
		'is_destination_of_has_source_is_source_of_has_destination_has_name': \
		['is_destination_of_has_source_is_source_of_has_destination_has_name'],
		'send_third': ['has_name', 'has_identifier_has_id_value']
	}

	number_of_classes = 15
	examples_per_class = 0

	total_actual = []
	total_predicted = []
	# accuracies = [0.4, 0.4, 0.4]
	sf_main = Storage_Files(data_folder, data_map_first)
	sf_sfm = Storage_Files(data_folder, data_map_sfm)
	sf_third = Storage_Files(data_folder, ['has_name', 'has_identifier_has_id_value'])
	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []
	for i in range(0, rounds):
		print("Fingerprint")
		# --- Fingerprint
		ccc = Column_Classification_Config()
		# Features
		ccc.add_feature('feature_main', 'Fingerprint', [sf_main, 15, 0, False, True])
		ccc.add_feature('feature_second', 'Corpus', [sf_sfm, 15, 0, False, True])
		ccc.add_feature('feature_third', 'Syntax_Feature_Model', [sf_third, 1, 0, False, False])
		

		# Matchers
		ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
		ccc.add_matcher('second', 'Word2Vec_Matcher', {'feature_second': 'corpus'}, ('matcher', 'send_second'))
		ccc.add_matcher('third', 'Syntax_Matcher', {'feature_third': 'syntax'}, ('second', 'send_third'))
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test_ckan(sm, test_folder, False)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp_acc.append(accuracy)
		tmp_prec.append(precision(actual, predicted))
		tmp_rec.append(recall(actual, predicted))
		tmp_fmeasure.append(f_measure(actual, predicted))

	accuracies.append( round(sum(tmp_acc) / float(rounds), 2) )
	precisions.append( round(sum(tmp_prec) / float(rounds), 2) )
	recalls.append( round(sum(tmp_rec) / float(rounds), 2) )
	f_measures.append(round(sum(tmp_fmeasure) / float(rounds), 2))
	gm.append_x("Pipeline")
	gm.append_y(accuracies)
	gm.append_y(precisions)
	gm.append_y(recalls)
	gm.append_y(f_measures)
	print(gm)
	
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_small_font(cm, classnames, normalize=True, title="Confusion Matrix Experiment 4 CKAN Dataset")

	gm.store(filename="/graph_maker/exp1.4ckan_b_1")
	subtitle = "Scores were averaged over " + str(rounds) + " tests"
	labels = ["Accuracy", "Precision", "Recall", "F-Measure"]
	gm.plot_bar_n("Improvement Rounds", "Scores", "Scores of Pipeline compared to Baseline", labels, subtitle=subtitle, rotation=0)


if __name__ == '__main__':
	# experiment4_inliers()
	experiment4_outliers()
