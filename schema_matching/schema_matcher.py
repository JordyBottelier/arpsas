
from .storage_files import Storage_Files
from .misc_func import *
from .feature_classes import *
from .column_classifiers import *
from .schema_reader import *

from sklearn.metrics import confusion_matrix, accuracy_score
from pandas_ml import ConfusionMatrix

class Schema_Matcher():
	"""
		Uses the column classification config to walk through the 
		matchers and calls the given method and handles the classification.

		When executing a tree method, the args should be a list of arguments.  
	"""
	def __init__(self, col_classification_config):
		self.matcher_tree = col_classification_config.get_tree()
		if self.matcher_tree.matcher == None:
			throw_error("Error, empty column classification config, no matchers found")
		self.schema_reader = Schema_Reader()

	def classify_column(self, args, detect_outlier=False):
		"""
			args should be a list and in that list should be a column and if detect_outlier is true, it should
			also contain that boolean
		"""
		object_method = 'classify_column'
		if detect_outlier:
			args.append(detect_outlier)
			return self.matcher_tree.classify(args, object_method, detect_outlier)
		else:
			prediction, outlier = self.matcher_tree.classify(args, object_method, detect_outlier)
			return prediction

	def classify_column_proba(self, args, detect_outlier=False):
		"""
			args should be a list and in that list should be a column and if detect_outlier is true, it should
			also contain that boolean
		"""
		object_method = 'classify_column_proba'
		if detect_outlier:
			args.append(detect_outlier)
			return self.matcher_tree.classify_proba(args, object_method, detect_outlier)
		else:
			prediction, certainty, outlier = self.matcher_tree.classify_proba(args, object_method, detect_outlier)
			return prediction, certainty, outlier

	def classify_instance_proba(self, args):
		"""
			args should be a list and in that list should be a string to classify
		"""
		object_method = 'classify_instance_proba'
		prediction, certainty, outlier = self.matcher_tree.classify_proba(args, object_method)
		return prediction, certainty

	def classify_instance(self, args):
		"""
			args should be a list and in that list should be a string to classify
		"""
		object_method = 'classify_instance'
		prediction, certainty = self.matcher_tree.classify(args, object_method)
		return prediction

	def execute_tests_matchers(self, num_tests=5, learnset_ratio=0.7):
		"""
			Perform a test on all of the matchers to see how well they classify their feature samples. 
		"""
		return self.matcher_tree.execute_test(num_tests, learnset_ratio)

	def classify_schema(self, path, min_avg_prob=0, detect_outlier=False):
		"""
			Classify an entire CSV schema, if min_avg_prob is set to a number higher than 0, than
			the probability classification scheme will be used, and if the average probability is lower
			than the min_avg_prob, than the result will be 'unknown'. If the min_avg_prob is set, the probabilities
			will also be added to the mapping. 
		"""
		columns = self.schema_reader.get_columns(path)
		mapping = {}
		certainty = 0
		count_number = 0
		for colname in columns:
			column = columns[colname]
			mapping[colname] = self.classify_schema_column(column, min_avg_prob, detect_outlier)
		return mapping

	def classify_schema_column(self, column, min_avg_prob=0, detect_outlier=False):
		"""
			Classify a column of a schema, use outlier detection and/or a minimal probability value. 

			I recommend that you use the min_avg_prob together with the outlier detector 
			(even if the min_avg_prob is very low) because this will trigger a more reliable outlier detection.
		"""
		result = None
		if min_avg_prob > 0:
			result = self.classify_column_proba([column], detect_outlier)
			certainty = result[1]
			if result[1] < min_avg_prob:
				result = ('unknown', result)
			elif detect_outlier and result[-1][0] > 0.5 and result[-1][0] > certainty:
				result = ('unknown', result)
			else:
				result = (result[0], result[1])
		else:
			result = self.classify_column([column], detect_outlier)
			if detect_outlier and result[-1] == 0:
				result = 'unknown'
			elif detect_outlier:
				result = result[0]
		return result

	def test_schema_matcher(self, columns, min_avg_prob=0, detect_outlier=False):
		"""
			The columns variable should be a list in which there are lists with column data. (use the schema reader
			to get them please using the get_duplicate_columns() method)

			It returns the predicted headers in a list, so they can be used for collecting results.
		"""
		result_headers = []
		i = 0
		for column in columns:
			i+=1
			result = self.classify_schema_column(column, min_avg_prob, detect_outlier)
			if min_avg_prob > 0:
				result_headers.append(result[0])
			else:
				result_headers.append(result)
		return result_headers


	def test_pipeline(self, sf, num_columns=10, examples_per_column=0, unique=False, use_map=False):
		"""
			Gather all the column data from a storage file object and get the accuracy. 
			This should be serperate data in comparison to your training data. 
		"""
		columns = []
		targets = []
		data = None
		if isinstance(sf, Storage_Files):
			dc = Data_Collector(sf, 
				num_columns=num_columns, examples_per_column=examples_per_column, unique=unique, use_map=use_map)
			data = dc.get_columns()
		elif type(sf) == dict:
			data=sf
		else:
			throw_error("Error bad data format")
		for entity in data:
				cols = data[entity]
				for col in cols:
					columns.append(col)
					targets.append(entity)
		results = []
		# run tests on data and print accuracy
		for i in range(0, len(columns)):
			col = columns[i]
			self.prep_columns(col)
			result = self.classify_column([col])
			results.append(result)
		labels = list(set(targets))
		print(ConfusionMatrix(targets, results))
		accuracy = accuracy_score(targets, results)
		print(accuracy)
		return accuracy

	def prep_columns(self, column):
		""" Make sure every element of a column is a string.
		"""
		for i in range(0, len(column)):
			column[i] = str(column[i])

