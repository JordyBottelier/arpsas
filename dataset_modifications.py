"""
	This file is mostly used for gathering graphs on the datasets and for modifying csv files in a folder to
	actually be used by the JORDY system. 
"""

from schema_matching import *
from schema_matching.misc_func import *
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
from os.path import isfile
import os
import math
import string

def get_letter(classname):
	"""
		For the ckan dataset, the classnames are very long, therefore we created a mapping to letters.
		This method is used to actually retrieve a letter based on the classname. 
	"""
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

"""
	Used to create the new, bigger, dataset -----------
"""
def prep_dataset(test_folder):
	"""
		From a folder of csv files, create folders with the classes serperated. Use this to create a learnset
		for the data collector from a folder with csv files. 
	"""
	sr = Schema_Reader()
	all_data = {}
	for filename in sorted(os.listdir(test_folder)):
		print(filename)

		path = test_folder + filename
		if(isfile(path)):
			headers, columns = sr.get_duplicate_columns(path)
			i = 0
			for header in headers:
				col = columns[i]
				col = remove_nan_from_list(col)
				col = remove_values_from_list(col, "")
				col = remove_values_from_list(col, " ")
				col = remove_values_from_list(col, "-         ")
				col = remove_values_from_list(col, "UNKNOWN")

				col = remove_values_from_list(col, 'UNKNOWN   ')
				if header in all_data:
					all_data[header] +=  col
				else:
					all_data[header] = col
				i += 1
	for classname in all_data:
		path = test_folder + classname
		if not os.path.exists(path):
			os.makedirs(path)
		data_path = path + "/" + classname + ".txt"
		data = all_data[classname]
		file = open(data_path,"w")
		file.write(str(data))

def prep_dataset_ckan(test_folder):
	"""
		From a folder of csv files, create folders with the classes serperated. Use this to create a learnset
		for the data collector from a folder with csv files. 
	"""
	sr = Schema_Reader()
	all_data = {}
	for filename in sorted(os.listdir(test_folder)):
		print(filename)

		path = test_folder + filename
		if(isfile(path)):
			headers, columns = sr.get_duplicate_columns(path)
			i = 0
			for header in headers:
				col = columns[i]
				col = remove_nan_from_list(col)
				col = remove_values_from_list(col, "")
				col = remove_values_from_list(col, " ")
				col = remove_values_from_list(col, "-         ")
				col = remove_values_from_list(col, "UNKNOWN")

				col = remove_values_from_list(col, 'UNKNOWN   ')
				if header in all_data:
					all_data[header] +=  col
				else:
					all_data[header] = col
				i += 1
	classname_map = {}
	classname_reverse = {} 
	alphabet = list(string.ascii_uppercase + string.ascii_lowercase)
	i = 0
	for classname in all_data:
		letter = alphabet[i]
		i += 1
		if len(classname) > 20:
			classname_map[letter] = classname
			classname_reverse[classname] = letter
		else:
			classname_map[classname] = classname
			classname_reverse[classname] = classname
		path = test_folder + classname
		if not os.path.exists(path):
			os.makedirs(path)
		data_path = path + "/" + classname + ".txt"
		data = all_data[classname]
		file = open(data_path,"w")
		file.write(str(data))

	classname_map['unknown'] = 'unknown'
	classname_reverse['unknown'] = 'unknown'
	print_dict(classname_reverse)
	store_file("ckan_subset/classname_map", classname_map)
	store_file("ckan_subset/classname_reverse", classname_reverse)


def prep_columns(column):
	""" Make sure every element of a column is a string.
	"""
	for i in range(0, len(column)):
		column[i] = str(column[i])

# Remove occuring values from a list
def remove_values_from_list(the_list, val):
	return [value for value in the_list if value != val]

# Remove value from list if substring is present (used for nan)
def remove_nan_from_list(the_list):
	return [value for value in the_list if not isNaN(value) ]

def isNaN(num):
	return num != num

"""
------------------------------------------------------------------
"""
		
def get_dataset_stats(test_folder, title="Company.Info Testset Statistics"):
	"""
		For a folder with csv files, read all the columns and plot how many columns there are
		and how many instances in total
	"""
	sr = Schema_Reader()
	headers_dict = {}
	for filename in sorted(os.listdir(test_folder)):
		print(filename)

		path = test_folder + filename
		if(isfile(path)):
			headers, columns = sr.get_duplicate_columns(path)
			i = 0
			for header in headers:
				col = columns[i]
				col = remove_nan_from_list(col)
				col = remove_values_from_list(col, "")
				col = remove_values_from_list(col, " ")
				col = remove_values_from_list(col, "-         ")
				col = remove_values_from_list(col, "UNKNOWN")
				col = remove_values_from_list(col, 'UNKNOWN   ')

				if header in headers_dict:
					headers_dict[header][0] += 1
					headers_dict[header][1] += len(col)
				else:
					headers_dict[header] = [1, len(col)]
				i += 1

	y1 = []
	y2 = []
	x = []
	for header in headers_dict:
		avg = round(headers_dict[header][1] / float(headers_dict[header][0]), 2)
		headers_dict[header].append(avg)
		x.append(header)
		y1.append(headers_dict[header][0])
		y2.append(headers_dict[header][1])

	print_dict(headers_dict)
	gm = Graph_Maker()
	gm.add_x(x)
	gm.append_y(y1)
	gm.append_y(y2)
	print(gm)
	gm.plot_bar_double_scale("Column Type", "Number of Occurences", "Total Number of Entries", 
		title, "Number of Occurences", "Total Number of Entries")


"""
------------------------------------------------------------------
"""
		
def get_dataset_stats_ckan(test_folder, title="Company.Info Testset Statistics"):
	"""
		For a folder with csv files, read all the columns and plot how many columns there are
		and how many instances in total, but use the ckan mapping to retrieve the correct letters for
		the classnames. 
	"""
	sr = Schema_Reader()
	headers_dict = {}
	for filename in sorted(os.listdir(test_folder)):
		print(filename)

		path = test_folder + filename
		if(isfile(path)):
			headers, columns = sr.get_duplicate_columns(path)
			i = 0
			for header in headers:
				col = columns[i]
				col = remove_nan_from_list(col)
				col = remove_values_from_list(col, "")
				col = remove_values_from_list(col, " ")
				col = remove_values_from_list(col, "-         ")
				col = remove_values_from_list(col, "UNKNOWN")
				col = remove_values_from_list(col, 'UNKNOWN   ')

				if header in headers_dict:
					headers_dict[header][0] += 1
					headers_dict[header][1] += len(col)
				else:
					headers_dict[header] = [1, len(col)]
				i += 1

	y1 = []
	y2 = []
	x = []
	for header in headers_dict:
		avg = round(headers_dict[header][1] / float(headers_dict[header][0]), 2)
		headers_dict[header].append(avg)
		x.append(get_letter(header))
		y1.append(headers_dict[header][0])
		y2.append(headers_dict[header][1])

	print_dict(headers_dict)
	gm = Graph_Maker()
	gm.add_x(x)
	gm.append_y(y1)
	gm.append_y(y2)
	print(gm)
	gm.plot_bar_double_scale("Column Type", "Number of Occurences", "Total Number of Entries", 
		title, "Number of Occurences", "Total Number of Entries")

def plot_learn_set():
	"""
		Simply plot the learnset, this is all the data that is in the data folders, didn't feel
		like creating something to read it all. 
	"""
	headers_dict = {   
		'address': 152539,
		'city': 180253,
		'company_name': 146991,
		'country': 95607,
		'date': 193302,
		'domain_name': 42557,
		'email': 33480,
		'gender': 15639,
		'house_number': 81807,
		'kvk_number': 16268,
		'legal_type': 18869,
		'person_name': 29530,
		'postcode': 176048,
		'province': 16953,
		'sbi_code': 28342,
		'sbi_description': 31448,
		'telephone_nr': 94419
	}
	gm = Graph_Maker()
	y1 = []
	x = []
	for header in headers_dict:
		x.append(header)
		y1.append(headers_dict[header])
	gm.add_x(x)
	gm.add_y(y1)
	gm.plot_bar("Column Type","Number of Occurences", "Company.Info Learnset Statistics")

def plot_learn_set_ckan():
	"""
		Simply plot the learnset, this is all the data that is in the data folders, didn't feel
		like creating something to read it all. 
	"""
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

	gm = Graph_Maker()
	y1 = []
	x = []
	for header in headers_dict:
		x.append(get_letter(header))
		y1.append(headers_dict[header])
	gm.add_x(x)
	gm.add_y(y1)
	gm.plot_bar("Column Type","Number of Occurences", "CERIF Learnset Statistics", rotation=90)


if __name__ == '__main__':
	plot_learn_set_ckan()
	plot_learn_set()
	get_dataset_stats('data_test/')
	test_folder = 'ckan_subset/testset/xml_csv/'
	get_dataset_stats_ckan(test_folder, title="CKAN Testset Statistics")

