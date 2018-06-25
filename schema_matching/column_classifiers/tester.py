
import numpy as np
from ..misc_func import *

class Tester():
	"""
		Class that can be used to test matcher classifiers
	"""

	def __init__(self):
		pass

	def k_fold_test_classifier(self, dataset_method, dataset_args, training_method, prediction_method, num_tests=5):
		""" 
			Runs k random tests.
			You have to specify the method that is needed to provide the classifier with the learn and testset datapoints.
			Pass the arguments as a list of ordered arguments needed by the function

			Make sure any method that gets a dataset to run in this test returns in the following order:
			learnset_datapoints, learnset_targetpoints, testset_datapoints, testset_targetpoints.

			the training method needs to train itself depending on the learnset_datapoints and learnset_targetpoints.

			The prediction method needs to be able to work with instances of the testset_datapoints, and needs to return 
			entities present in testset_targetpoints

			These points need to be pre-processed.
		"""
		results = []
		for i in range(0, num_tests):
			learnset_datapoints, learnset_targetpoints, testset_datapoints, testset_targetpoints \
			 = dataset_method(*dataset_args)
			
			acc = self.get_results_single_test(learnset_datapoints, learnset_targetpoints, testset_datapoints, testset_targetpoints, \
					training_method, prediction_method)
			results.append(acc)
		accuracy = sum(results) / float(num_tests)
		# print("Accuracy: " + str(accuracy))
		return accuracy


	def get_results_single_test(self, learnset_datapoints, learnset_targetpoints, testset_datapoints, \
			testset_targetpoints, training_method, prediction_method):
		
		""" Train the model and test the accuracy """
		training_method(learnset_datapoints, learnset_targetpoints)
		good = 0
		bad = 0
		for n in range(0, len(testset_datapoints)):
			prediction = prediction_method(testset_datapoints[n])
			if prediction in testset_targetpoints[n]:
				good += 1
			else:
				bad += 1
		return (good/(float(bad) + float(good)))

	def create_datasets(self, datapoints, targetpoints, learnset_ratio):
		""" Split the dataset in a test and learn set 
			kwargs should be: datapoints, targetpoints, learnset_ratio
		"""
		learnset_datapoints = []
		testset_datapoints = []
		learnset_targetpoints = []
		testset_targetpoints = []
		for n in range(0, len(datapoints)):
			pick = np.random.uniform()
			if pick < learnset_ratio:
				learnset_datapoints.append(datapoints[n])
				learnset_targetpoints.append(targetpoints[n])
			else:
				testset_datapoints.append(datapoints[n])
				testset_targetpoints.append(targetpoints[n])
		return learnset_datapoints, learnset_targetpoints, testset_datapoints, testset_targetpoints

	def k_fold_test_incremental_random(self, dataset_method, dataset_args, training_method, 
		prediction_method, num_tests=5):
		"""
			Run num_tests tests. For each test we randomly pick n classes that are part of the test.
			After each iteration we add 1 more class to the entire process, so we can get an overview of 
			the accuracy loss of each added class. 
		"""
		_, learnset_targetpoints, _, testset_targetpoints= dataset_method(*dataset_args)
		total_accuracy = []
		classsize, ratio = self.get_class_size(learnset_targetpoints, testset_targetpoints)
		print(classsize)
		print(ratio)
		xdata = []
		ydata = []
		for n in range(2, len(list(set(learnset_targetpoints))) + 1):
			"""
				First loop used to add an extra class to each iteration. 
			"""
			results = []
			for i in range(0, num_tests):
				"""
					Second loop used to perform a random test with the given amount of classes 
				"""
				learnset_datapoints, learnset_targetpoints, testset_datapoints, testset_targetpoints \
				= dataset_method(*dataset_args) # Get the datapoints randomly

				# Get the classes we are testing this iteration
				random_classes = self.get_random_classes(learnset_targetpoints, n) 

				# Filter learn and testset based upon the random classes
				learnset_datapoints, learnset_targetpoints = self.filter_datasets(random_classes, \
					learnset_datapoints, learnset_targetpoints)
				testset_datapoints, testset_targetpoints = self.filter_datasets(random_classes, \
					testset_datapoints, testset_targetpoints)

				# Get the accuracy result
				acc = self.get_results_single_test(learnset_datapoints, learnset_targetpoints, testset_datapoints, testset_targetpoints, \
						training_method, prediction_method)
				results.append(acc)
				xdata.append(n)
				ydata.append(acc)
			accuracy = sum(results) / float(num_tests)
			total_accuracy.append(accuracy)
			print("Accuracy: " + str(accuracy))
		text = "Number of test per iteration: " + str(num_tests) + \
		"\nNumber of examples per class per test: " + str(classsize) + \
		"\nLearnset ratio: " + str(ratio)
		plot_scatter(xdata, ydata, "Number of Classes", "Accuracy", "K-Fold Tests per Classes", "syntax_feature_test", True, True, text)
		return total_accuracy

	def get_class_size(self, learnset_targetpoints, testset_targetpoints):
		classname = learnset_targetpoints[0]
		a = learnset_targetpoints.count(classname)
		b = testset_targetpoints.count(classname)
		return a + b, a / float(a + b)

	def get_random_classes(self, targetpoints, num_points):
		"""
			pick num_points random classes from a list
		"""
		p = list(set(targetpoints))
		np.random.shuffle(p)
		return p[0:num_points]

	def filter_datasets(self, classes, datapoints, targetpoints):
		"""
			Only return data and targetpoints that exist in the classes-list, do this by looking up 
			the classes in the targetpoints and appending these data and targetpoints to a new list
		"""
		data_result = []
		target_result = []
		for i in range(0, len(targetpoints)):
			target = targetpoints[i]
			if target in classes:
				target_result.append(targetpoints[i])
				data_result.append(datapoints[i])
		return data_result, target_result