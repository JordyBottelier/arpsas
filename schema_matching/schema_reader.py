import pandas as pd
from .misc_func import *
import cchardet as chardet

class Schema_Reader():
	"""
		Read schemas and return them
	"""

	def __init__(self):
		pass

	def get_columns(self, source_path):
		rawdata = open(source_path, 'rb').read()
		result = chardet.detect(rawdata)
		charenc = result['encoding']
		csv = pd.read_csv(source_path, error_bad_lines=False, encoding = charenc)
		cols = {}
		for colname in csv:
			col = csv[colname]
			col = self.prep_columns(col)
			if len(col) > 0:
				cols[colname] = col
		return cols

	def get_duplicate_columns(self, source_path, skip_unknown=False):
		"""
			Read csv schemas and allow duplicate column names (used for testing the framework)
		"""
		rawdata = open(source_path, 'rb').read()
		result = chardet.detect(rawdata)
		charenc = result['encoding']
		delimiter = self.get_delimiter(source_path, charenc)
		columns = pd.read_csv(source_path, error_bad_lines=False, encoding = charenc, \
			skiprows=[0], header=None, delimiter=delimiter, warn_bad_lines=False, dtype=str)
		headers = pd.read_csv(source_path, error_bad_lines=True, encoding = charenc,\
		 header=None, nrows=1, delimiter=delimiter, warn_bad_lines=False)

		result = []
		result_headers = []
		for col_num in columns:
			header = headers[col_num][0]
			col = columns[col_num]
			col = self.prep_columns(col)
			if len(col) > 0:
				if skip_unknown and header != 'unknown':
					result.append(col)
					result_headers.append(header)
				elif not skip_unknown:
					result.append(col)
					result_headers.append(header)
		return result_headers, result

	def prep_columns(self, column):
		""" Make sure every element of a column is a string.
		"""
		result = []
		for i in range(0, len(column)):
			if not self.isNaN(column[i]) and column[i] != " " and column[i] != "":
				result.append(str(column[i]))
		return result

	def remove_nan_from_list(self, the_list):
		""" Remove nan from list """
		return [value for value in the_list if not self.isNaN(value) ]

	def isNaN(self, num):
		""" Check if number is NaN """
		return num != num

	def get_delimiter(self, source_path, charenc):
		"""
			Dumb function to check if the delimiter is a ; or a ,.
			We simply read the header with , as a delimiter and check the amount of ; that occur.
			If its more than 5 than ; is probably the delimter. 
		"""
		headers = pd.read_csv(source_path, error_bad_lines=True, encoding = charenc, header=None, nrows=1, delimiter=",", warn_bad_lines=False)
		if str(headers[0][0]).count(";") > 3:
			return ";"
		else:
			return ","