"""
	This file is used to prepare the CKAN-SERIF dataset. 

	it contains methods to transform both the RDF trees and the XML trees to json, and then
	parse them to a csv format. 
"""


from schema_matching import *
from schema_matching.misc_func import *
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
import os
from os.path import exists
from xmljson import parker as bf
from xml.etree.ElementTree import fromstring
import json
import sys
import shutil
import pandas as pd
import numpy as np
import copy
import collections
import sys

def get_json_xml(data_folder, num_entries=0):
	"""
		Using elemtree, parse xml to json using the parker transformation.

		The parker transformation keeps the tree structure in the json but removes the attributes. 
		There are however no attributes in the xml.
	"""
	i = 0
	result_dict = {}
	for filename in sorted(os.listdir(data_folder)):
		target_filename = data_folder + filename
		f = open(target_filename, 'r')
		text = f.read()
		xml = fromstring(text)
		mydict = json.loads(json.dumps(bf.data(xml)))
		result_dict[filename[0:-4]] = mydict
		i += 1
		if i == num_entries:
			break
	return result_dict


def flatten_json_recursive(entry, cumulated_key=''):
	"""
		Recursively loop through the json dictionary and flatten it out in a column structure. 
		For every value in the tree, we use the path to the value as a key. This method returns
		2 lists, 1 with keys and 1 with values, which later can be accumulated to form colums
	"""
	csv_keys = []
	csv_values = []
	for key in entry:
		value = entry[key]
		key_name = cumulated_key + key
		if 'key' in entry and 'value' in entry: # In case there is literally a dictionary with key and values
			item_key = entry['key']
			item_value = entry['value']
			return [item_key], [item_value]
		# If we encounter a value, add it to the list, as well as the path to the value
		elif type(value) == int or type(value) == bool or type(value) == float or type(value) == str:
			csv_keys.append(key_name)
			csv_values.append(value)
		# If we encounter another dictionary we also have to flatten it out. 
		# therefore we have to call the recursive method again. 
		elif type(value) == dict:
			result_keys, result_values = flatten_json_recursive(value, key_name + "_")
			csv_keys += result_keys
			csv_values += result_values
		# if we encounter a list it could be that we find duplicate values, therefore we ha
		elif type(value) == list:
			i = 0
			for list_item in value:
				try:
					result_keys, result_values = flatten_json_recursive(list_item, key_name + "_" + str(i) + "_")
					csv_keys += result_keys
					csv_values += result_values
					i += 1
				except TypeError:
					pass

	return csv_keys, csv_values

def redivide_set(data_folder_source, data_folder_target):
	"""
		This method is used to divide the complete datasets in two sets,
		a learnset and a testset. 
	"""
	target_files = os.listdir(data_folder_target)
	for i in range(0, len(target_files)):
		target_files[i] = target_files[i][0:-4]

	for filename in os.listdir(data_folder_source):
		if filename[0:-4] in target_files:
			total_path = data_folder_source + "/" + filename
			shutil.copy2(total_path, '../ckan_subset/source2/' + filename)
			print(filename)

def split_learn_test(source_xml, source_rdf, target, learnset_size=0.7):
	"""
		Split the rdf and xml files randomly in a learn and testset
	"""
	learnset_folder = "learnset/"
	testset_folder = "testset/"
	xml_folder = "xml/"
	rdf_folder = "rdf/"
	xml_files = os.listdir(source_xml)
	rdf_files = os.listdir(source_rdf)
	for i in range(0, len(xml_files)):
		base_filename = xml_files[i][0:-4]
		source_xml_file = source_xml + xml_files[i]
		source_rdf_file = source_rdf + base_filename + ".rdf"
		xml_target = ""
		rdf_target = ""
		if np.random.random_sample() < learnset_size: # append to learnset
			xml_target = target + learnset_folder + xml_folder + xml_files[i]
			rdf_target = target + learnset_folder + rdf_folder + base_filename + ".rdf"
		
		else: # append to testset
			xml_target = target + testset_folder + xml_folder + xml_files[i]
			rdf_target = target + testset_folder + rdf_folder + base_filename + ".rdf"
		shutil.copy2(source_xml_file, xml_target)
		shutil.copy2(source_rdf_file, rdf_target)



def fill_dict_cols(collection):
		"""
			Convert the collection, which is a dictionary in which the keys are column names
			and the values are the columns
		"""
		result = {}
		max_length = 0
		for col in collection:
			result[col] = collection[col]
			if len(collection[col]) > max_length:
				max_length = len(collection[col])
		
		# fill the list with empty entries so we can use the pandas dataframe
		for res in result:
			col = result[res]
			addition = max_length - len(col)
			col += [""] * addition
		return result

def fill_list_cols(collection):
	"""
		Fill the lists with empty items so pandas can store it
	"""
	result = []
	max_length = 0
	for col in collection:
		result.append(col)
		if len(col) > max_length:
			max_length = len(col)

	for col in result:
		addition = max_length - len(col)
		col += [""] * addition
	return result

def to_csv(collection, store_file):
	collection = fill_dict_cols(collection)
	df = pd.DataFrame(collection)
	csv = df.to_csv(store_file, index=False)

def to_csv_headers_columns(headers, columns, store_file):
	"""
		Go from a collection of csv headers and columns to a pandas datafame and store it. 
	"""
	columns = fill_list_cols(columns)
	df = pd.DataFrame(columns)
	df = df.transpose()
	df.columns = headers
	csv = df.to_csv(store_file, index=False)



def match_score(col1, col2):
	"""
		Compute the match score between two columns. We use the overlapping elements as a 
		scoring system. Even the slightest overlap is good enough
	"""
	matching_items = 0
	for item in list(set(col1)):
		for item2 in col2:
			if item == item2 or str(item).lower() == str(item2).lower():
				matching_items += 1
	return matching_items

def match_columns(data_folder_xml, data_folder_rdf, num_entries=200):
	"""
		Create a mapping for the columns in two collections, we use this to
		finally match the rdf dataset to the csv dataset.
	"""
	xml_dicts = get_json_xml(data_folder_xml, num_entries)
	rt = RDFTransformer()
	print("parsing rdf")
	rdf_dicts = rt.get_dictionaries(data_folder_rdf, num_entries)
	collection_xml = {}
	collection_rdf = {}
	print("creating collection")
	for title in xml_dicts:
		rdf_keys, rdf_vals = flatten_json_recursive(rdf_dicts[title])
		xml_keys, xml_vals = flatten_json_recursive(xml_dicts[title]['result'])
		add_collection(rdf_keys, rdf_vals, collection_rdf)
		add_collection(xml_keys, xml_vals, collection_xml)
	print("cleaning")
	collection_xml = clean_collection(collection_xml)
	collection_rdf = clean_collection(collection_rdf)

	print("getting mapping")
	matching = {}
	for xml_key in collection_xml:
		xml_col = collection_xml[xml_key]
		matching[xml_key] = []
		for rdf_key in collection_rdf:
			rdf_col = collection_rdf[rdf_key]
			score = match_score(rdf_col, xml_col)
			if score != 0:
				matching[xml_key].append((rdf_key, score))
				matching[xml_key] = sorted(matching[xml_key], key=lambda tup: tup[1], reverse=True)

	print_dict(matching)
	headers = []
	columns = []
	mapping = {}
	for xml_key in matching:
		if len(matching[xml_key]) > 0:
			new_header = matching[xml_key][0][0]
			headers.append(new_header)
			mapping[xml_key] = new_header
		else:
			mapping[xml_key] = 'unknown'
			headers.append('unknown')
		columns.append(collection_xml[xml_key])
	print_dict(mapping)
	store_file("../ckan_subset/xml_rdf_map", mapping)
	to_csv_headers_columns(headers, columns, "../ckan_subset/target/xml_source_parsed.csv")
	to_csv(collection_rdf, "../ckan_subset/target/rdf_target.csv")



def add_collection(keys, values, collection):
	"""
		Add values to the dictionary
	"""
	for i in range(0, len(keys)):
		key = keys[i]
		value = values[i]
		if key not in collection:
			collection[key] = [value]
		else:
			collection[key].append(value)

def remove_value_from_list(x, val):
	return [n for n in x if n != val]

def prep_col(column):
	"""
		Remove unwanted values from the columns
	"""
	tmp = copy.deepcopy(column)
	tmp = remove_value_from_list(tmp, "null")
	tmp = remove_value_from_list(tmp, None)
	tmp = remove_value_from_list(tmp, "")
	tmp = remove_value_from_list(tmp, '')
	return tmp

def clean_collection(collection, num_entries=50):
	"""
		Prepare the collection for a transformation to csv.
		Unwanted values are removed here. 
		If there are less then num_entries in the collection, do not keep the column.
		
	"""	
	result = {}
	collection = concatenate_list_headers(collection)
	for col in collection:
		new_col = prep_col(collection[col])
		if len(new_col) > num_entries:
			result[col] = new_col
	return result

def concatenate_list_headers(cols):
	"""
		The json flattening causes list items to be concatenated with an underscore and unique identifier.
		But we would like to add columns with the same identifier together. 
	"""
	result = {}
	for head in cols:
		entries = head.split("_")
		index = hasInt(entries)
		if index != 0:
			del entries[index : index + 1]
			new_head = "_".join(entries)
			if new_head in result:
				result[new_head] += cols[head]
			else:
				result[new_head] = cols[head]
		else:
			result[head] = cols[head]
	return result


def hasInt(myList):
	index = 0
	for n in myList:
		if n.isdigit():
			return index
		index += 1
	return 0


def create_csv_set_xml(folder, target_folder, mapfile="../ckan_subset/xml_rdf_map", num_per_col=50):
	"""
		create csv files from the xml files by flattening them into columns.
		Num_per_col = number of xml files parsed to 1 csv
	"""
	mapping = read_file(mapfile)
	xml_dicts = get_json_xml(folder, 0)
	collection = {}
	i = 0
	filename = 0
	for title in xml_dicts:
		xml_keys, xml_vals = flatten_json_recursive(xml_dicts[title]['result'])
		add_collection(xml_keys, xml_vals, collection)
		i += 1
		if i == num_per_col:
			titles = []
			headers = []
			columns = []
			filename += 1
			i = 0
			collection = clean_collection(collection, 20) # prevent occurence of very little values
			for col in collection:
				try:
					headers.append(mapping[col])
				except:
					headers.append("unknown")
				columns.append(collection[col])
			target_filename = target_folder + str(filename) + ".csv"
			to_csv_headers_columns(headers, columns, target_filename)
			collection = {}
	# For the left overs
	headers = []
	columns = []
	filename += 1
	collection = clean_collection(collection)
	for col in collection:
		try:
			headers.append(mapping[col])
		except:
			headers.append("unknown")
		columns.append(collection[col])
	target_filename = target_folder + str(filename) + ".csv"
	to_csv_headers_columns(headers, columns, target_filename)
	collection = {}

def create_csv_set_rdf(folder, target_folder, num_per_col=50):
	"""
		create csv files from the rdf files by flattening them into columns.
		(we first parse to json and then flatten)
		Num_per_col = number of rdf files parsed to 1 csv
	"""
	rt = RDFTransformer()
	rdf_dicts = rt.get_dictionaries(folder, 0)
	collection = {}
	i = 0
	filename = 0
	for title in rdf_dicts:
		rdf_keys, rdf_vals = flatten_json_recursive(rdf_dicts[title])
		add_collection(rdf_keys, rdf_vals, collection)
		i += 1
		if i == num_per_col:
			filename += 1
			i = 0
			collection = clean_collection(collection, num_entries=20)
			headers = []
			columns = []
			for col in collection:
				headers.append(col)
				columns.append(collection[col])
			target_filename = target_folder + str(filename) + ".csv"
			print(target_filename)
			to_csv_headers_columns(headers, columns, target_filename)
			collection = {}

	# For the left overs
	headers = []
	columns = []
	filename += 1
	collection = clean_collection(collection)
	for col in collection:
		headers.append(mapping[col])
		columns.append(collection[col])
	target_filename = target_folder + str(filename) + ".csv"
	to_csv_headers_columns(headers, columns, target_filename)
	collection = {}



if __name__ == '__main__':

	data_folder_rdf = '../ckan_subset/target2/'
	data_folder_xml = '../ckan_subset/source2/'
	# match_columns(data_folder_xml, data_folder_rdf, 0)
	create_csv_set_xml('../ckan_subset/learnset/xml/', '../ckan_subset/learnset/xml_csv/', num_per_col=50)
	create_csv_set_xml('../ckan_subset/testset/xml/', '../ckan_subset/testset/xml_csv/', num_per_col=50)

	create_csv_set_rdf('../ckan_subset/learnset/rdf/', '../ckan_subset/learnset/rdf_csv/', num_per_col=100)
	create_csv_set_rdf('../ckan_subset/testset/rdf/', '../ckan_subset/testset/rdf_csv/', num_per_col=100)
