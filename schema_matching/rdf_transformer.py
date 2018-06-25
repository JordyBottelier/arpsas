import rdflib
import sys
import os


class RDFTransformer():
	"""
		Class to transform a folder with rdf to json or csv
	"""

	def __init__(self):
		pass

	def get_graph(self, data_folder, num_entries=0):
		"""
			Turn a folder of rdfs into a graph using rdflib
		"""
		i = 0
		g = rdflib.Graph()
		g.open("store", create=True)
		for filename in os.listdir(data_folder):
			i += 1
			target_filename = data_folder + filename
			g.parse(target_filename)
			if i == num_entries:
				break
		return g

	def get_dictionaries(self, data_folder, num_entries=0):
		""" 
			Turn a folder of rdf files into graphs and then into dictionaries
			by unpacking the rdf tree. 
		"""
		i = 0
		result_dict = {}
		for filename in sorted(os.listdir(data_folder)):
			g = rdflib.Graph()
			g.open("store", create=True)
			target_filename = data_folder + filename
			g.parse(target_filename)
			for primary in self.get_primaries(g):
				result_dict[filename[0:-4]] = self.unpack_entry(g, primary)
			i += 1
			if i == num_entries:
				break
		return result_dict

	def get_predicate(self, entry):
		entry = str(entry)
		return entry.split("#")[-1]

	def get_value(self, entry):
		entry = str(entry)
		if "#" in entry:
			return entry.split("#")[-1]
		return entry

	def unpack_entry(self, g, entry):
		"""
			Loop recursively through the tree of triples and store everything in a dictionary
		"""
		result = {}
		predicate_obs = list(g.predicate_objects(entry))
		# If we are not dealing with an entity type (connection type) we append the object
		if len(predicate_obs) == 0:
			return self.get_value(entry)
		else:
			# loop through the subjects and unpack them
			for pred, obj in predicate_obs:
				# saveguard against self relations
				if obj != entry:
					result[self.get_predicate(pred)] = self.unpack_entry(g, obj)
				else:
					result[self.get_predicate(pred)] = self.get_value(entry)
		return result

	def get_csv(self, entry_dict, cumulated_key=''):
		"""
			loop recursively through a dictionary return a list of keys/values
			where the keys are the paths to the values in the dictionary
		"""
		csv_keys = []
		csv_values = []
		for key in entry_dict:
			# An actual csv value
			key_name = key.replace("has_", '')
			key_name = cumulated_key + key_name
			if ('has_' in key or 'label' in key or 'type' in key) and type(entry_dict[key]) != dict:
				csv_keys.append(key_name)
				csv_values.append(entry_dict[key])
			elif 'has_' in key and type(entry_dict[key]) == dict:
				result_keys, result_values = self.get_csv(entry_dict[key], key_name + "_")
				csv_keys += result_keys
				csv_values += result_values
		return csv_keys, csv_values

	def get_primaries(self, g):
		"""
			Get all the primary entries by recursively going up in the tree. 
		"""
		subjects, predicates, objects = self.get_subj_pred_obj(g)
		objects = list(set(objects))
		predicates = list(set(predicates))
		subjects = list(set(subjects))
		primaries = []
		for subject in subjects:
				prim = self.get_primary_entry(g, subject)
				primaries.append(prim)
		primaries = list(set(primaries))
		return primaries

	def get_primary_entry(self, g, entry):
		"""
			Get the primary subjects by looping through the tree
		"""
		subs_preds = list(g.subject_predicates(entry))
		if len(subs_preds) == 0:
			return entry
		else:
			for subject, predicate in subs_preds:
				# saveguard against self relations
				if subject != entry:
					return self.get_primary_entry(g, subject)
		return entry


	def get_subj_pred_obj(self, g):
		"""
			Get all the triple items
		"""
		subjects = []
		objects = []
		predicates = []
		for subject, predicate, object in g:
			predicates.append(predicate)
			subjects.append(subject)
			objects.append(object)
		return subjects, predicates, objects
