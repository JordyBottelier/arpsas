"""
Baseline experiment to test the performance of each matcher on the dataset.

The code in all experiment files is very ugly, but it I did not see the use in creating beautifull
code for all experiments since it is a 'press enter and rerun them' setup
"""


from schema_matching import *
from schema_matching.misc_func import *
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from os.path import isfile
import os
import math


def execute_test(sm, test_folder, skip_unknown=False):
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
		try:
			if(isfile(path)):
				headers, columns = sr.get_duplicate_columns(path, skip_unknown)
				actual += headers
				result_headers = None
				if skip_unknown:
					result_headers = sm.test_schema_matcher(columns, 0, False)
				else:
					result_headers = sm.test_schema_matcher(columns, 0.4, True)
				predicted += result_headers
		except:
			print("Fail")
		# break
	# print(ConfusionMatrix(actual, predicted))
	return actual, predicted

def experiment1_inliers():
	data_folder = 'data_train/'
	gm = Graph_Maker()
	gm.store()
	rounds = 5
	x = ["Fingerprint", "Syntax Feature Model", "Word2Vec Matcher"]
	
	number_of_classes = 50
	examples_per_class = 0
	accuracies = []
	total_actual = []
	total_predicted = []
	# accuracies = [0.4, 0.4, 0.4]
	classes = ['address', 'city', 'company_name', 'country', 'date', 'domain_name', 'email', 'gender', 'house_number',\
	'kvk_number', 'legal_type', 'person_name', 'postcode', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
	sf_main = Storage_Files(data_folder, classes)
	tmp = []
	for i in range(0, rounds):
		print("Fingerprint")
		# --- Fingerprint
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Fingerprint', [sf_main, number_of_classes, examples_per_class, False, False])
		ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	accuracy = round(sum(tmp) / float(rounds), 2)
	accuracies.append(accuracy)

	classnames = get_class_names(total_actual)
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True)
	
	tmp = []
	total_actual = []
	total_predicted = []
	for i in range(0, rounds):
		print("SFM")
		# --- Syntax Feature Model
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Syntax_Feature_Model', [sf_main, 1, 5000, False, False])

		ccc.add_matcher('matcher', 'Syntax_Matcher', {'feature_main': 'syntax'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	accuracy = round(sum(tmp) / float(rounds), 2)
	accuracies.append(accuracy)

	classnames = get_class_names(total_actual)
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True)

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
		actual, predicted = execute_test(sm, 'data_test/', True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	accuracy = round(sum(tmp) / float(rounds), 2)
	accuracies.append(accuracy)

	classnames = get_class_names(total_actual)
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True)

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
	data_folder = 'data_train/'
	gm = Graph_Maker()
	gm.store()
	rounds = 5
	x = ["Fingerprint", "Syntax Feature Model", "Word2Vec Matcher"]
	
	number_of_classes = 50
	examples_per_class = 0
	accuracies = []
	precisions = []
	recalls = []
	fmeasures = []

	classes = ['address', 'city', 'company_name', 'country', 'date', 'domain_name', 'email', 'gender', 'house_number',\
	'kvk_number', 'legal_type', 'person_name', 'postcode', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
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
		actual, predicted = execute_test(sm, 'data_test/', False)
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
	classnames = get_class_names(total_actual)
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True)
	
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
		ccc.add_feature('feature_main', 'Syntax_Feature_Model', [sf_main, 1, 5000, False, False])

		ccc.add_matcher('matcher', 'Syntax_Matcher', {'feature_main': 'syntax'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', False)
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
	classnames = get_class_names(total_actual)
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True)

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
		actual, predicted = execute_test(sm, 'data_test/', False)
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
	classnames = get_class_names(total_actual)
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True)

	gm.add_x(x)
	# accuracies = [0.4, 0.4, 0.4]
	# precisions = [0.5, 0.5, 0.5]
	# recalls = [0.62, 0.62, 0.62]
	# fmeasures = [0.23, 0.23, 0.28]
	gm.append_y(accuracies)
	gm.append_y(precisions)
	gm.append_y(recalls)
	gm.append_y(fmeasures)
	subtitle = "Scores were averaged over " + str(rounds) + " tests with " + str(len(classes)) + " classes. " + \
	"Number of simulated columns per class: " + str(number_of_classes)
	labels = ["Accuracy", "Precision", "Recall", "F-Measure"]
	gm.plot_bar_n("Matcher Type", "Score", "Accuracy of Matchers", labels, subtitle=subtitle)

def confusion_number_matcher():
	classes = ['telephone_nr', 'house_number']
	data_folder = 'data_train/'
	gm = Graph_Maker()
	gm.store()
	rounds = 1
	number_of_classes = 100
	accuracies = []
	total_actual = []
	total_predicted = []
	# accuracies = [0.4, 0.4, 0.4]
	sf_main = Storage_Files(data_folder, classes)
	tmp = []
	for i in range(0, rounds):
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Number_Feature', [sf_main, number_of_classes, 0, False, False])
		ccc.add_matcher('matcher', 'Number_Matcher', {'feature_main': 'number_feature'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True)
		total_actual += actual
		total_predicted += predicted
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	# Filter out the non-used classes
	result_total = []
	result_pred = []
	for i in range(0, len(total_actual)):
		if total_actual[i] in classes:
			result_total.append(total_actual[i])
			result_pred.append(total_predicted[i])
	# accuracy = round(sum(tmp) / float(rounds), 2)
	# accuracies.append(accuracy)
	total_actual = result_total
	total_predicted = result_pred

	classnames = get_class_names(total_actual)
	cm = confusion_matrix(total_actual, total_predicted, labels=classnames)
	gm.plot_confusion_matrix(cm, classnames, normalize=True)

def get_class_names(ytrue):
	res = []
	for c in ytrue:
		if c not in res:
			res.append(c)
	return res

if __name__ == '__main__':
	# experiment1_inliers()
	# experiment1_outliers()
	confusion_number_matcher()