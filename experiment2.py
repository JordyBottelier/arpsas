"""
Number of columns vs scores experiment.

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

rounds = 1
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
	# print(ConfusionMatrix(actual, predicted))
	return actual, predicted

def experiment2_inliers():
	data_folder = 'data_train/'
	gm = Graph_Maker()
	x = [20, 40, 60, 80, 100, 120]
	
	max_number_of_columns = 121
	step_size = 20
	min_number_of_columns = 20
	examples_per_class = 0
	accuracies = []
	# accuracies = [0.4, 0.4, 0.4]
	classes = ['city', 'country', 'date', 'gender', 'house_number',\
	'legal_type', 'postcode', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
	sf_main = Storage_Files(data_folder, classes)
	tmp = []
	for i in range(min_number_of_columns, max_number_of_columns, step_size):
		print("Fingerprint")
		print(i)
		accs = []
		for l in range(0, rounds):
			# --- Fingerprint
			ccc = Column_Classification_Config()
			ccc.add_feature('feature_main', 'Fingerprint', [sf_main, i, examples_per_class, False, False])

			ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
			sm = Schema_Matcher(ccc)
			actual, predicted = execute_test(sm, 'data_test/', True)
			accuracy = accuracy_score(actual, predicted)
			accs.append(accuracy)
		tmp.append(accuracy)
	gm.append_y(tmp)

	tmp = []
	for i in range(min_number_of_columns, max_number_of_columns, step_size):
		print("W2V")
		print(i)
		# --- Word2Vec Matcher
		accs = []
		for l in range(0, rounds):
			ccc = Column_Classification_Config()
			ccc.add_feature('feature_main', 'Corpus', [sf_main, i, examples_per_class, False, False])

			ccc.add_matcher('matcher', 'Word2Vec_Matcher', {'feature_main': 'corpus'}) # main classifier
			sm = Schema_Matcher(ccc)
			actual, predicted = execute_test(sm, 'data_test/', True)
			accuracy = accuracy_score(actual, predicted)
			accs.append(accuracy)
		tmp.append(round(sum(accs) / float(rounds), 2))
	gm.append_y(tmp)

	# gm.append_y([0.6, 0.7, 0.8, 0.9, 1])
	# gm.append_y([0.6, 0.5, 0.3, 0.2, 1])
	gm.add_x(x)
	gm.store(filename="/graph_maker/exp1.2a")
	gm.plot_line_n("Number of Columns", "Accuracy", "Accuracy vs Number of Columns" ,["Fingerprint", "Word2Vec Matcher"])


def experiment2_outliers():
	"""
		Run a full experiment on all matchers including outliers and
		measure precision, recall, f-measure and accuracy
	"""
	data_folder = 'data_train/'
	gm = Graph_Maker()
	x = [20, 40, 60, 80, 100, 120]
	
	max_number_of_columns = 121
	step_size = 20
	min_number_of_columns = 20
	examples_per_class = 0
	accuracies = []
	precisions = []
	recalls = []
	fmeasures = []

	classes = ['city', 'country', 'date', 'gender', 'house_number',\
	'legal_type', 'postcode', 'province', 'sbi_code', 'sbi_description', 'telephone_nr']
	sf_main = Storage_Files(data_folder, classes)
	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []

	for i in range(min_number_of_columns, max_number_of_columns, step_size):
		print("Fingerprint")
		# --- Fingerprint
		accs = []
		precs = []
		recs = []
		fmeass = []
		for l in range(0, rounds):

			ccc = Column_Classification_Config()
			ccc.add_feature('feature_main', 'Fingerprint', [sf_main, i, examples_per_class, False, False])

			ccc.add_matcher('matcher', 'Fingerprint_Matcher', {'feature_main': 'fingerprint'}) # main classifier
			sm = Schema_Matcher(ccc)
			actual, predicted = execute_test(sm, 'data_test/', False)
			acc = accuracy_score(actual, predicted)
			prec = precision(actual, predicted)
			rec = recall(actual, predicted)
			fmeas = f_measure(actual, predicted)
			accs.append(acc)
			precs.append(prec)
			recs.append(rec)
			fmeass.append(fmeas)
		tmp_acc.append(round(sum(accs) / float(rounds), 2))
		tmp_prec.append(round(sum(precs) / float(rounds), 2))
		tmp_rec.append(round(sum(recs) / float(rounds), 2))
		tmp_fmeasure.append(round(sum(fmeass) / float(rounds), 2))

	gm.append_y(tmp_acc)
	gm.append_y(tmp_prec)
	gm.append_y(tmp_rec)
	gm.append_y(tmp_fmeasure)

	tmp_acc = []
	tmp_prec = []
	tmp_rec = []
	tmp_fmeasure = []
	for i in range(min_number_of_columns, max_number_of_columns, step_size):
		print("W2V")
		accs = []
		precs = []
		recs = []
		fmeass = []
		for l in range(0, rounds):
			# --- Word2Vec Matcher
			ccc = Column_Classification_Config()
			ccc.add_feature('feature_main', 'Corpus', [sf_main, i, examples_per_class, False, False])

			ccc.add_matcher('matcher', 'Word2Vec_Matcher', {'feature_main': 'corpus'}) # main classifier
			sm = Schema_Matcher(ccc)
			actual, predicted = execute_test(sm, 'data_test/', False)
			acc = accuracy_score(actual, predicted)
			prec = precision(actual, predicted)
			rec = recall(actual, predicted)
			fmeas = f_measure(actual, predicted)
			accs.append(acc)
			precs.append(prec)
			recs.append(rec)
			fmeass.append(fmeas)
		tmp_acc.append(round(sum(accs) / float(rounds), 2))
		tmp_prec.append(round(sum(precs) / float(rounds), 2))
		tmp_rec.append(round(sum(recs) / float(rounds), 2))
		tmp_fmeasure.append(round(sum(fmeass) / float(rounds), 2))

	gm.append_y(tmp_acc)
	gm.append_y(tmp_prec)
	gm.append_y(tmp_rec)
	gm.append_y(tmp_fmeasure)


	gm.add_x(x)
	# gm.append_y([0.4, 0.4, 0.4, 0.4, 0.4])
	# gm.append_y([0.5, 0.5, 0.5, 0.5, 0.5])
	# gm.append_y([0.62, 0.62, 0.62, 0.34, 0.74])
	# gm.append_y([0.23, 0.23, 0.28, 0.21, 0.24])
	# gm.append_y([0.47, 0.49, 0.44, 0.6, 0.8])
	# gm.append_y([0.95, 0.55, 0.35, 0.14, 0.56])
	# gm.append_y([0.61, 0.42, 0.53, 0.43, 0.57])
	# gm.append_y([0.28, 0.27, 0.46, 0.56, 0.86])
	gm.store(filename="/graph_maker/exp1.2b")
	labels = [ "Fingerprint - Accuracy",
	"Fingerprint - Precision",
	"Fingerprint - Recall",
	"Fingerprint - F-Measure",
	"Word2Vec Matcher - Accuracy",
	"Word2Vec Matcher - Precision",
	"Word2Vec Matcher - Recall",
	"Word2Vec Matcher - F-Measure",
	]
	gm.plot_line_n("Number of Columns", "Score", "Score vs Number of Columns" ,labels)

if __name__ == '__main__':
	experiment2_inliers()
	experiment2_outliers()