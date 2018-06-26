
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


classes = os.listdir("ckan_subset/prepared_learnset/")

def get_letter(classname):
	mapping = read_file("ckan_subset/classname_reverse")
	letter_to_ckan = read_file("ckan_subset/classname_map")
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
				result_headers = sm.test_schema_matcher(columns, 0.4, True)
			for head in result_headers:
				predicted.append(get_letter(head))
			for head in headers:
				actual.append(get_letter(head))
		# except:
		# 	print("Fail")
		# break
	# print(ConfusionMatrix(actual, predicted))
	return actual, predicted

def experiment1_inliers():
	data_folder = "ckan_subset/prepared_learnset/"
	test_folder = 'ckan_subset/testset/xml_csv/'
	gm = Graph_Maker()
	gm.store()
	rounds = 5
	x = ["Fingerprint", "Syntax Feature Model", "Word2Vec Matcher"]
	
	number_of_classes = 15
	examples_per_class = 0
	accuracies = []
	total_actual = []
	total_predicted = []
	# accuracies = [0.4, 0.4, 0.4]
	sf_main = Storage_Files(data_folder, classes)
	tmp = []
	for i in range(0, rounds):
		print("Fingerprint")
		# --- Fingerprint
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Fingerprint', [sf_main, number_of_classes, examples_per_class, False, False])
		ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test_ckan(sm, test_folder, True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	accuracy = round(sum(tmp) / float(rounds), 2)
	accuracies.append(accuracy)
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_small_font(cm, classnames, normalize=True)
	
	tmp = []
	total_actual = []
	total_predicted = []
	for i in range(0, rounds):
		print("SFM")
		# --- Syntax Feature Model
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Syntax_Feature_Model', [sf_main, 1, 0, False, False])

		ccc.add_matcher('matcher', 'Syntax_Matcher', {'feature_main': 'syntax'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test_ckan(sm, test_folder, True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	accuracy = round(sum(tmp) / float(rounds), 2)
	accuracies.append(accuracy)
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_small_font(cm, classnames, normalize=True)

	tmp = []
	total_actual = []
	total_predicted = []
	for i in range(0, rounds):
		print("W2V")
		# --- Word2Vec Matcher
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Corpus', [sf_main, number_of_classes, examples_per_class, False, False])

		ccc.add_matcher('matcher', 'Word2Vec_Matcher', {'feature_main': 'corpus'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test_ckan(sm, test_folder, True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	accuracy = round(sum(tmp) / float(rounds), 2)
	accuracies.append(accuracy)
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_small_font(cm, classnames, normalize=True)
	

	print(accuracies)
	gm.add_x(x)
	gm.add_y(accuracies)
	subtitle = "Accuracy was averaged over " + str(rounds) + " tests with " + str(len(classes)) + " classes. " + \
	"Number of simulated columns per class: " + str(number_of_classes)
	gm.plot_bar("Matcher Type", "Accuracy", "Accuracy of Matchers", subtitle=subtitle, show_value=True)


def experiment1_outliers():
	"""
		Run a full experiment on all matchers including outliers and
		measure precision, recall, f-measure and accuracy
	"""
	data_folder = "ckan_subset/prepared_learnset/"
	test_folder = 'ckan_subset/testset/xml_csv/'
	gm = Graph_Maker()
	gm.store()
	rounds = 5
	x = ["Fingerprint", "Syntax Feature Model", "Word2Vec Matcher"]
	
	number_of_classes = 15
	examples_per_class = 0
	accuracies = []
	precisions = []
	recalls = []
	fmeasures = []
	sf_main = Storage_Files(data_folder, classes)
	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []
	total_actual = []
	total_predicted = []

	for i in range(0, rounds):
		print("Fingerprint")
		# --- Fingerprint
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Fingerprint', [sf_main, number_of_classes, examples_per_class, False, False])

		ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
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
	fmeasures.append(round(sum(tmp_fmeasure) / float(rounds), 2))
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	#gm.plot_confusion_matrix(cm, classnames, normalize=True)
	
	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []
	total_actual = []
	total_predicted = []
	for i in range(0, rounds):
		print("SFM")
		# --- Syntax Feature Model
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Syntax_Feature_Model', [sf_main, 1, 0, False, False])

		ccc.add_matcher('matcher', 'Syntax_Matcher', {'feature_main': 'syntax'}) # main classifier
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
	fmeasures.append(round(sum(tmp_fmeasure) / float(rounds), 2))
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	#gm.plot_confusion_matrix(cm, classnames, normalize=True)

	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []
	total_actual = []
	total_predicted = []
	for i in range(0, rounds):
		print("W2V")
		# --- Word2Vec Matcher
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Corpus', [sf_main, number_of_classes, examples_per_class, False, False])

		ccc.add_matcher('matcher', 'Word2Vec_Matcher', {'feature_main': 'corpus'}) # main classifier
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
	fmeasures.append(round(sum(tmp_fmeasure) / float(rounds), 2))
	classnames = list(set(get_class_names(total_actual) + get_class_names(total_predicted)))
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	#gm.plot_confusion_matrix(cm, classnames, normalize=True)

	gm.add_x(x)
	# accuracies = [0.4, 0.4, 0.4]
	# precisions = [0.5, 0.5, 0.5]
	# recalls = [0.62, 0.62, 0.62]
	# fmeasures = [0.23, 0.23, 0.28]
	gm.append_y(accuracies)
	gm.append_y(precisions)
	gm.append_y(recalls)
	gm.append_y(fmeasures)
	gm.store()
	subtitle = "Scores were averaged over " + str(rounds) + " tests with " + str(len(classes)) + " classes. " + \
	"Number of simulated columns per class: " + str(number_of_classes)
	labels = ["Accuracy", "Precision", "Recall", "F-Measure"]
	gm.plot_bar_n("Matcher Type", "Score", "Accuracy of Matchers", labels, subtitle=subtitle)


def get_class_names(ytrue):
	res = []
	for c in ytrue:
		if c not in res:
			res.append(c)
	return res

if __name__ == '__main__':
	experiment1_inliers()
	experiment1_outliers()
