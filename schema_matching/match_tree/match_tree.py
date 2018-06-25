
from ..misc_func import *
import operator
import numpy as np

class Match_Tree():
	"""
		Matcher_name: Name of the matcher that is going to be added. 
		submatcher_name: Name of the matcher of which the new matcher will be a child. 
		submatcher_class: classname for which the submatcher will be called when classifying. 
		(if the classname is classA and the parent classifies a column as classA, then the new submatcher
		will be called upon the data and that classification result will be used)
	"""

	def __init__(self):
		self.matcher_names = []
		self.matcher = None	# actual matcher object. 
		self.matcher_name = ""
		# a dict with submatcher_names and a list of all their submatchers (for searching the tree)
		self.submatchers_submatcherslist = {} 
		self.submatcher_names = {} # submatcher name -> classname for which submatcher is called
		self.parent_classes = {} # class name -> submatcher name (if this class is predicted, call submatcher X)
		self.submatchers = {} # submatchers name -> Match_Tree()

	def add_matcher_to_tree(self, matcher_name, matcher, parent_classes=('root', None)):
		"""
			Validate the new entry and add it to the current branch or send the request to a submatcher
		"""
		validation = self.validate_entry(matcher_name, matcher, parent_classes)
		parent_name = None
		submatcher_class = None
		if type(validation) == bool:
			# This is the new root node, so the matcher has been added
			return True
		else:
			parent_name = validation[0]
			submatcher_class = validation[1]

		# This is the parent node so we have to create a new submatcher entry
		if parent_name == self.matcher_name:
			self.add_matcher_to_current_branch(matcher_name, matcher, submatcher_class)
			return True

		else:
			# look in the tree where the submatcher has to be inserted
			for submatcher in self.submatchers_submatcherslist:
				submatcherslist = self.submatchers_submatcherslist[submatcher]
				# The parent of the new matcher is a submatcher of this current node or one of its children,
				# so we add the node to the parent.
				# We also add the new matcher to the submatchers list
				if parent_name == submatcher or parent_name in submatcherslist:
					self.submatchers[submatcher].add_matcher_to_tree(matcher_name, matcher, parent_classes)
					submatcherslist.append(matcher_name)
					return True
			return False

	def add_matcher_to_current_branch(self, matcher_name, matcher, submatcher_class):
		"""
			Add a submatcher to this tree leaf. 
			1. Create a new Match_tree object.
			2. Add the submatcher to the tree as a root class.
			3. Add the new submatcher to all the dictionary and lists
		"""
		if submatcher_class in self.parent_classes:
			errorstring = "Error, double declaration for class " \
				+ submatcher_class + " for matcher " + self.matcher_name
			throw_error(errorstring)

		self.submatchers[matcher_name] = Match_Tree()
		self.submatchers[matcher_name].add_matcher_to_tree(matcher_name, matcher)
		self.submatchers_submatcherslist[matcher_name] = []
		self.submatcher_names[matcher_name] = submatcher_class
		self.parent_classes[submatcher_class] = matcher_name


	def validate_entry(self, matcher_name, matcher, parent_classes):
		"""
			Check if the matcher names arent used double or if the parent matcher for a submatcher is missing.
			Possibly set the root.
		"""
		if matcher_name in self.matcher_names:
			print("Warning, matcher name already exists in tree, \
				this might cause unexpected behaviour/overwrites. You might want to use a different name")

		self.matcher_names.append(matcher_name)
		if parent_classes == ('root', None):
			self.matcher_name = matcher_name
			self.matcher = matcher
			return True
		elif type(parent_classes) != tuple:
			throw_error("Error, parent_classes argument of wrong type")
		elif parent_classes[1] == None:
			throw_error("Bad class for submatcher")

		submatcher_name = parent_classes[0]
		submatcher_class = parent_classes[1]

		if submatcher_name not in self.matcher_names:
			errorstring = "Error, (sub)matcher: \'" + submatcher_name + \
				"\'' not in tree so this matcher can not be added"
			throw_error(errorstring)

		return submatcher_name, submatcher_class

	def get_repr(self):
		my_dict = {}
		my_dict['matcher_name'] = self.matcher_name
		my_dict['submatcher_classes'] = self.submatcher_names
		for submatcher in self.submatchers:
			tree_obj = self.submatchers[submatcher]
			my_dict[submatcher] = tree_obj.get_repr()
		return my_dict

	def print_tree(self):
		print_dict(self.get_repr())

	def execute_test(self, num_tests=5, learnset_ratio=0.7):
		""" 
			walk the tree and execute the execute_test method on the children with the given argument
		"""
		result = self.matcher.execute_test(num_tests, learnset_ratio)
		print("Matcher " + self.matcher_name + ": " + str(result))
		for sub_name in self.submatchers:
			matcher = self.submatchers[sub_name]
			matcher.execute_test(num_tests, learnset_ratio)


	def classify(self, args, classification_method, detect_outlier=False):
		command = "result = self.matcher." + classification_method + "(*args)"
		loc = {'args': args,
				'self': self}
		exec(command, globals(), loc)
		result = loc['result']
		outlier = 1
		if detect_outlier:
			# If we use the outlier detection 
			outlier = int(result[1][0])
			result = result[0]

		if type(result) == list or type(result) == type(np.array([])):
			result = result[0]
		if result in self.parent_classes:
			selected_submatcher_name = self.parent_classes[result]
			sub_tree_obj = self.submatchers[selected_submatcher_name]
			result, outlier = sub_tree_obj.classify(args, classification_method, detect_outlier)
			return result, outlier
		else:
			return result, outlier

	def classify_proba(self, args, classification_method, detect_outlier=False):
		"""
			Use the matchers probability classification functions (which should return a list in which a list 
			of probabilities per class is found), and return the average probability of the selected class
		"""
		command = "result = self.matcher." + classification_method + "(*args)"
		loc = {'args': args,
				'self': self}
		exec(command, globals(), loc)
		result = loc['result']
		outlier = 1
		if detect_outlier:
			outlier = result[1][0]
			result = result[0]
		classes = self.matcher.get_classes()
		if type(result) == list or type(result) == type(np.array([])):
			result = result[0]
		result_dict = dict(zip(classes, result))
		prediction = max(result_dict.items(), key=operator.itemgetter(1))[0]
		chance = result_dict[prediction]
		if prediction in self.parent_classes:
			selected_submatcher_name = self.parent_classes[prediction]
			sub_tree_obj = self.submatchers[selected_submatcher_name]
			new_pred, cum_chance, outlier = sub_tree_obj.classify_proba(args, classification_method, detect_outlier)
			return new_pred, (cum_chance * chance), outlier
		else:
			return prediction, chance, outlier

