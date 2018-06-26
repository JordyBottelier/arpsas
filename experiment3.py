"""
Number of instances/column vs scores experiment.

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
				print("done")
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

def experiment3_inliers():
	data_folder = 'data_train/'
	gm = Graph_Maker()
	x = [30, 60, 90, 120, 150, 0]
	
	number_of_columns = 100
	accuracies = []
	# accuracies = [0.4, 0.4, 0.4]
	classes = ['city', 'country', 'date', 'gender', 'house_number',\
	'legal_type', 'postcode', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
	sf_main = Storage_Files(data_folder, classes)
	tmp = []
	for i in x:
		print("Fingerprint")
		print(i)
		# --- Fingerprint
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Fingerprint', [sf_main, number_of_columns, i, False, False])

		ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True)
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	gm.append_y(tmp)

	tmp = []
	for i in x:
		print("SFM")
		print(i)
		# --- Fingerprint
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Syntax_Feature_Model', [sf_main, 1, i * 100, False, False])

		ccc.add_matcher('matcher', 'Syntax_Matcher', {'feature_main': 'syntax'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True)
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	gm.append_y(tmp)

	tmp = []
	for i in x:
		print("W2V")
		print(i)
		# --- Word2Vec Matcher
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Corpus', [sf_main, number_of_columns, i, False, False])

		ccc.add_matcher('matcher', 'Word2Vec_Matcher', {'feature_main': 'corpus'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', True)
		accuracy = accuracy_score(actual, predicted)
		tmp.append(accuracy)
	gm.append_y(tmp)

	# gm.append_y([0.6, 0.7, 0.6, 0.8, 0.7, 0.9])
	# gm.append_y([0.6, 0.5, 0.3, 0.4, 0.5, 0.7])
	x = [30, 60, 90, 120, 150, 180]
	xticks = [30, 60, 90, 120, 150, 0]
	gm.add_x(x)
	gm.store(filename="/graph_maker/exp1.3a")
	gm.plot_line_n("Number of Instances per Column", "Accuracy", "Accuracy vs Number of Instances per Column" ,["Fingerprint",\
	"Syntax Feature Model Matcher","Word2Vec Matcher"], xticks=xticks)


def experiment3_outliers():
	"""
		Run a full experiment on all matchers including outliers and
		measure precision, recall, f-measure and accuracy
	"""
	data_folder = 'data_train/'
	gm = Graph_Maker()
	x = [30, 60, 90, 120, 150, 0]
	
	number_of_columns = 100
	min_number_of_columns = 20
	examples_per_class = 0

	classes = ['city', 'country', 'date', 'gender', 'house_number',\
	'legal_type', 'postcode', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
	sf_main = Storage_Files(data_folder, classes)
	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []

	for i in x:
		print("Fingerprint")
		# --- Fingerprint
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Fingerprint', [sf_main, number_of_columns, i, False, False])

		ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', False)
		accuracy = accuracy_score(actual, predicted)
		tmp_acc.append(accuracy)
		tmp_prec.append(precision(actual, predicted))
		tmp_rec.append(recall(actual, predicted))
		tmp_fmeasure.append(f_measure(actual, predicted))
	gm.append_y(tmp_acc)
	gm.append_y(tmp_prec)
	gm.append_y(tmp_rec)
	gm.append_y(tmp_fmeasure)

	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []

	for i in x:
		print("SFM")
		# --- Fingerprint
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Syntax_Feature_Model', [sf_main, 1, i * 100, False, False])

		ccc.add_matcher('matcher', 'Syntax_Matcher', {'feature_main': 'syntax'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', False)
		accuracy = accuracy_score(actual, predicted)
		tmp_acc.append(accuracy)
		tmp_prec.append(precision(actual, predicted))
		tmp_rec.append(recall(actual, predicted))
		tmp_fmeasure.append(f_measure(actual, predicted))

	gm.append_y(tmp_acc)
	gm.append_y(tmp_prec)
	gm.append_y(tmp_rec)
	gm.append_y(tmp_fmeasure)

	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []
	for i in x:
		print("W2V")
		# --- Word2Vec Matcher
		ccc = Column_Classification_Config()
		ccc.add_feature('feature_main', 'Corpus', [sf_main, number_of_columns, i, False, False])

		ccc.add_matcher('matcher', 'Word2Vec_Matcher', {'feature_main': 'corpus'}) # main classifier
		sm = Schema_Matcher(ccc)
		actual, predicted = execute_test(sm, 'data_test/', False)
		accuracy = accuracy_score(actual, predicted)
		tmp_acc.append(accuracy)
		tmp_prec.append(precision(actual, predicted))
		tmp_rec.append(recall(actual, predicted))
		tmp_fmeasure.append(f_measure(actual, predicted))

	gm.append_y(tmp_acc)
	gm.append_y(tmp_prec)
	gm.append_y(tmp_rec)
	gm.append_y(tmp_fmeasure)


	# gm.add_x(x)
	x = [30, 60, 90, 120, 150, 180]
	gm.append_x(x)
	gm.append_x(x)
	gm.append_x(x)
	# gm.append_y([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
	# gm.append_y([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	# gm.append_y([0.62, 0.62, 0.62, 0.34, 0.74, 0.62])
	# gm.append_y([0.23, 0.23, 0.28, 0.21, 0.24, 0.24])
	# gm.append_y([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
	# gm.append_y([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	# gm.append_y([0.62, 0.62, 0.62, 0.34, 0.74, 0.62])
	# gm.append_y([0.23, 0.23, 0.28, 0.21, 0.24, 0.24])
	# gm.append_y([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
	# gm.append_y([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
	# gm.append_y([0.62, 0.62, 0.62, 0.34, 0.74, 0.62])
	# gm.append_y([0.23, 0.23, 0.28, 0.21, 0.24, 0.24])
	gm.store(filename="/graph_maker/exp1.3b")
	subtitles = ["Fingerprint", "Syntax Feature Model Matcher", "Word2Vec Matcher"]
	labels = ["Accuracy", "Precision", "Recall", "F-Measure"]
	xticks = [30, 60, 90, 120, 150, 0]
	gm.subplot_n("Number of Instances per Column", "Scores", "Scores vs Number of Instances per Column" ,subtitles, labels*3, xticks=xticks)

if __name__ == '__main__':
	# experiment3_inliers()
	# experiment3_outliers()
	gm = Graph_Maker()
	gm.load(filename="/graph_maker/exp1.3a")
	xticks = [30, 60, 90, 120, 150, 0]
	gm.plot_line_n("Number of Instances per Column", "Accuracy", "Accuracy vs Number of Instances per Column" ,["Fingerprint",\
	"Syntax Feature Model Matcher","Word2Vec Matcher"], xticks=xticks)