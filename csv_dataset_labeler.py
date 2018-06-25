"""
	This file was used to manually but quickly label all the source csv files. 
	It shows a schema, asks if you actually want to use this schema (you can discard it if it is not usable). 
	
	It then asks per column if you want to use/discard the column, and what the new column name should be. 
	All csv files are then stored in a folder. 
"""
import os
from schema_matching import *
import sys
import pandas as pd
from subprocess import call
from os.path import isfile
from io import StringIO

class Dataset_Labeler():
	"""
		Manually label a dataset.

		We track the used classes and ask for confirmation if you want to add a new one, we do this because 
		mistakes can happen, and thats okay. 
	"""

	def __init__(self, path_to_batcher="../batchtool/cases", storage="all_csv_labeled/", path_to_input="/input/"):
		self.path_to_batcher = path_to_batcher
		self.storage_folder = storage
		self.path_to_input = path_to_input
		self.used_cases_filename = "used_cases"
		self.classes_filename = "classes"
		self.sr = Schema_Reader()
		try:
			self.classes = read_file(storage + "system/" + self.classes_filename)
		except:
			self.classes = ['city', 'postcode', 'kvk_number', 'address', 'company_name', 
			'email', 'telephone_nr', 'domain_name', 'bik', 'legal_type', 'sbi_code', 'house_number', 'addition']

		try:
			self.used_cases = read_file(storage + "system/" + self.used_cases_filename)
		except:
			self.used_cases = []

	def label_dataset(self):
		"""
			Loop through the folders and get the cases
		"""
		for foldername in os.listdir(self.path_to_batcher):
			if foldername not in self.used_cases:
				path_to_batch_input = self.path_to_batcher + "/" + foldername + self.path_to_input
				try:
					self.handle_schema_folder(path_to_batch_input, foldername)
				except FileNotFoundError:
					print("Input not found for: ")
					print(foldername)


	def handle_schema_folder(self, path_to_batch_input, batch_name):
		"""
			Manually label a case and store it
		"""
		for filename in os.listdir(path_to_batch_input):
			if filename[-4::].lower() == ".csv":
				schema_path = path_to_batch_input + filename
				try:
					headers, columns = self.get_schema(schema_path)
					use_schema = self.show_schema(self.storage_folder + "system/tmp.csv")
					if use_schema:
						headers, columns = self.modify_schema(headers, columns)
						self.store_schema(headers, columns, batch_name)
				except UnicodeDecodeError:
					pass
		self.used_cases.append(batch_name)
		self.used_cases = list(set(self.used_cases))
		store_file(self.storage_folder + "system/" + self.used_cases_filename, self.used_cases)
		store_file(self.storage_folder + "system/" + self.classes_filename, self.classes)



	def modify_schema(self, headers, columns):
		"""
			Classify every column of the schema, and return the new headers with their columns. 
			Check if the class you inputted was has already been used, otherwise add it. 
		"""
		new_headers = []
		new_columns = []
		for i in range(0, len(headers)):
			column = columns[i]
			header = headers[i]
			print_list(reversed(column))
			print("Header: " + str(header))
			print("length: " + str(len(column)))
			print(self.classes)
			while True:
				text = input("What will the new header name be?\n")
				if text in self.classes:
					new_headers.append(text)
					new_columns.append(column)
					break
				elif text == 's':
					break
				else:
					text2 = input("Class not in classes, was it a mistake or do you want to add it? (a)\n")
					if text2 == 'a':
						self.classes.append(text)
						new_headers.append(text)
						new_columns.append(column)
						break
		return new_headers, new_columns

	def show_schema(self, path):
		"""
			Show and validate schema
		"""
		command = "tad " + path
		call(command, shell=True)
		text = ""
		while True:
			text = str(input("Do you want to use this schema? (y/n)\n"))
			if text == 'y' or text == 'n':
				break
			print("Wrong input, try again")
		if text == 'y':
			return True
		elif text == 'n':
			return False

	def get_schema(self, schema_path):
		"""
			Read the input schema
		"""
		headers, columns = self.sr.get_duplicate_columns(schema_path)
		col_dict = self.convert_to_dict(columns)
		df = pd.DataFrame(col_dict)
		
		csv = df.to_csv(header=headers, index=False)
		# Temporarily store the schema for viewing purposes
		store_file(self.storage_folder + "system/tmp.csv", csv)
		return headers, columns

	def convert_to_dict(self, columns):
		"""
			Convert the list of columns to a dictionary so pandas can store it
		"""
		result = {}
		max_length = 0
		for i in range(0, len(columns)):
			result[i] = columns[i]
			if len(columns[i]) > max_length:
				max_length = len(columns[i])
		
		# fill the list with empty entries so we can use the pandas dataframe
		for res in result:
			col = result[res]
			addition = max_length - len(col)
			col += [""] * addition
		return result

	def store_schema(self, headers, columns, batch_name):
		"""
			Store the classified schema and check in the list if the batch name hasnt already been used.
		"""
		col_dict = self.convert_to_dict(columns)
		df = pd.DataFrame(col_dict)
		filename = self.get_filename(batch_name)
		csv = df.to_csv(self.storage_folder + filename, header=headers, index=False)
		# store_file(self.storage_folder + filename, csv)


	def get_filename(self, batch_name):
		"""
			We store the schemas as batch_name_number.csv,
			here we extract the number for the schema and return the new filename
		"""
		max_number = -1
		for filename in os.listdir(self.storage_folder):
			if filename.endswith(".csv"):
				filename = filename[0:-4]
				number = int(filename.split("_")[0])
				if number > max_number:
					max_number = number
		max_number += 1
		return str(max_number) + "_" + batch_name + ".csv"

def modify_file():
	"""
		test method to see if we can drop the row numbers
	"""
	name = "all_csv_labeled/"
	filename = "test.csv"
	# for filename in os.listdir(name):
	if filename.endswith(".csv"):
		print(filename)
		csv = read_file(name + filename)
		t = StringIO(csv)
		df = pd.read_csv(t, sep=",", header=None)
		df = df.rename(columns=df.iloc[0], copy=False).iloc[1:].reset_index(drop=True)
		df.drop(0, axis=0)
		print(list(df))
		df.to_csv(name + "x" + filename, index=False)
		sys.exit(0)
		# df = pd.DataFrame.from_csv(name+filename)


if __name__ == '__main__':
	dl = Dataset_Labeler()
	# modify_file()
	dl.label_dataset()