from .misc_func import *

class Storage_Files():
	"""
		Minimal class used to store paths to the data folders and the classes.
		Example of valid datamap:

		data_map = {
			'place': ['address', 'city', 'postcode'],
			'numbers': ['sbi_code', 'kvk_number', 'bik'],
			'name': ['company_name'],
			'contact': ['telephone_nr', 'email']
		}
		The keys will be used as classes and the list items are the folders in which they can be found
	"""


	def __init__(self, data_folder, classes=[]):
		self.data_folder = data_folder
		
		if type(classes) == list and len(classes) > 1:
			self.classes = classes
		elif type(classes) == dict and len(classes) > 1:
			self.data_map = classes
			self.classes = list(classes.keys())
		else:
			errorstring = "Classes should be defined in a list structure or datamap, and should have at least 2 elements"
			throw_error(errorstring)

	def get_map(self):
		return self.data_map

	def get_classes(self):
		return self.classes

	def get_data_folder(self):
		return self.data_folder

