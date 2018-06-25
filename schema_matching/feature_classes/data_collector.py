
from ..misc_func import *
import os
import ast
import random

class Data_Collector():
	"""
		Needs het storage files object to get all the classes and path to the data.
		If you want to overwrite the way data is gathered, this is the place to do it.
		
		Behaviour:
		You always state how many columns you want per class of your total data. You can use all of the data
		(examples_per_column=0), or you can specify how many samples of the data you want per column. 

		The data is gathered according to the class names or the names in the datamap. 
		The folders should contain files with a readable python list. 
		
	"""

	def __init__(self, sf=None, num_columns=1, examples_per_column=0, unique=False, use_map=False):
		"""
			num_columns: Total number of columns you want to simulate.
			examples_per_column: Boundary to the total amount of entries you want to use from your data
		"""
		self.sf = sf
		self.classes = sf.get_classes()
		self.data_folder = sf.get_data_folder()
		self.columns = {}
		if num_columns < 1:
			throw_error("Must have at least one column per class")
		self.columns = self.get_columns_chunks(num_columns, examples_per_column, unique=unique, use_map=use_map)


	def get_columns_chunks(self, num_columns, examples_per_column, unique=False, use_map=False):
		"""
			Get the column data but split them into chunks, so you can treat every chunk as a data entry.
			If the number of examples per column is 0, we split on a best effor basis. 
		"""
		columns = self.get_columns_bare(unique=unique, use_map=use_map)
		if examples_per_column > 0:
			self.guard_number_examples(columns, num_columns, examples_per_column)
		col_stats = self.get_column_stats(columns)
		stats = {}
		for entity in columns:
			column = columns[entity]
			chunks = []
			if examples_per_column == 0:
				random.shuffle(column)
				# Get the total number of chunks using all entries
				chunks = [column[i::num_columns] for i in range(num_columns)]
			else:
				chunks = self.get_chunks(column, num_columns, examples_per_column)
			for chunk in chunks:
				if entity not in stats:
					stats[entity] = len(chunk)
				else:
					stats[entity] += len(chunk)

			# Check if there are no empty entries
			if [] in chunks:
				errorstring = ("Error: amount of sample data you have is smaller than number of columns you want to generate," +
					" either get a larger dataset or reduce the number of examples you want in your data")
				throw_error(errorstring)
			columns[entity] = chunks
		
		if use_map:
			"""
				Get the data back together according to the mapping.
			"""
			columns = self.reassemble(columns)

		""" Print the amount of entries of the chunks: """
		if examples_per_column == 0:
			print("Average amount of examples per class")
			for s in stats:
				stats[s] = stats[s] / float(num_columns)
			print_dict(stats)
		else:
			print("Total examples per class:")
			print_dict(stats)
		print("Number of columns per class: " + str(num_columns))
		print("Number of examples per column: " + str(examples_per_column))

		return columns

	def reassemble(self, columns):
		"""
			Move the chunks to the correct entity of the datamap. 
		"""
		new_columns = {}
		datamap = self.sf.get_map()
		for entity in datamap:
			new_columns[entity] = []
			for folder in datamap[entity]:
				new_columns[entity] += columns[folder]
		return new_columns

	def get_chunks(self, column, num_columns, examples_per_column):
		""" 
			Split a column randomly into chunks using the total number of examples
			We already know this is possible because of the check done earlier.

			We need to fill num_columns with num_examples random examples.
		"""
		examples_per_column = examples_per_column
		chunks = []
		random.shuffle(column)
		start = 0
		end = examples_per_column
		for i in range(0, num_columns):
			chunks.append(column[start:end])
			start = end + 1
			end = end + examples_per_column + 1
		return chunks


	def get_columns_map(self, unique=False):
		""" 
			Read data from all the folders that are in the datamap classes and concatenate them.
			The map is a dict with classes with a list of folders with the data that should be read for the class.
		"""
		datamap = self.sf.get_map()
		columns = {}
		for entity in datamap:
			for folder in datamap[entity]:
				columns[folder] = []
				destination_folder = self.data_folder + folder + "/"
				for filename in os.listdir(destination_folder):
					file = open(destination_folder + "/" + filename, "r")
					contents = file.read()
					content_list = ast.literal_eval(contents)
					columns[folder] += list(content_list)
		if unique:
			for entity in self.classes:
				columns[entity] = list(set(columns[entity]))

		print("Total entries:")
		col_stats = self.get_column_stats(columns)
		print_dict(col_stats)
		return columns

	def get_columns_bare(self, unique=False, use_map=False):
		""" Read all of the files that come along with 
			the classes and collect their data, put them in normal columns """
		columns = {}
		if use_map:
			columns = self.get_columns_map(unique)
		else:
			for entity in self.classes:
				columns[entity] = []
				destination_folder = self.data_folder + entity + "/"
				for filename in os.listdir(destination_folder):
					file = open(destination_folder + "/" + filename, "r")
					contents = file.read()
					content_list = ast.literal_eval(contents)
					columns[entity] += list(content_list)
			if unique:
				for entity in self.classes:
					columns[entity] = list(set(columns[entity]))

			print("Total entries:")
			col_stats = self.get_column_stats(columns)
			print_dict(col_stats)

		return columns

	def guard_number_examples(self, columns, num_columns, examples_per_column):
		"""
			Check if the amount of needed samples is not bigger than the amount of samples you have
		"""
		smallest = 10000000
		for entity in columns:
			if len(columns[entity]) < smallest:
				smallest = len(columns[entity])

		if (num_columns * examples_per_column) > smallest:
			throw_error("Impossible with this dataset to have so many columns with this amount of examples")

	def get_column_stats(self, columns):
		""" Print the amount of entries in each entity column """
		result = {}
		for entity in columns:
			result[entity] = len(columns[entity])
		return result

	def get_column_stats_set(self, columns):
		""" Print the amount of entries in each entity column """
		result = {}
		for entity in columns:
			result[entity] = len(set(columns[entity]))
		return result

	def get_columns(self):
		return self.columns

