"""
	Schema matching example test using company.info data
"""
from schema_matching import *
from schema_matching.misc_func import *
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
from os.path import isfile
import os

def execute_test(sm, test_folder, skip_unknown=False):
	sr = Schema_Reader()
	accuracies = []
	total_input_headers = []
	total_output_headers = []
	for filename in sorted(os.listdir(test_folder)):
		print(filename)
		path = test_folder + filename
		if(isfile(path)):
			try:
				headers, columns = sr.get_duplicate_columns(path, skip_unknown)
				result_headers = None
				if skip_unknown:
					result_headers = sm.test_schema_matcher(columns, 0, False)
				else:
					result_headers = sm.test_schema_matcher(columns, 0.4, True)
				total_output_headers += result_headers
				total_input_headers += headers
				print(list(zip(headers, result_headers)))
				accuracy = accuracy_score(headers, result_headers)
				print(accuracy)
				accuracies.append(accuracy)
			except:
				print("fail")
		break
	print(accuracies)
	print(accuracy_score(total_input_headers, total_output_headers))
	print(len(total_input_headers))
	print(ConfusionMatrix(total_input_headers, total_output_headers))

		

if __name__ == '__main__':
	data_map_main = {
		'text': ['address', 'city', 'company_name', 'country', 'gender',\
			'legal_type', 'person_name', 'province', 'sbi_description'],
		'date_tel': ['date', 'telephone_nr'],
		'email': ['email'],
		'domain_name': ['domain_name'],
		'postcode': ['postcode'],
		'numbers': ['kvk_number', 'sbi_code', 'house_number']
	}
	data_map_numbers = {
		'kvk_sbi': ['kvk_number', 'sbi_code'],
		'telephone_nr': ['telephone_nr'],
		'house_number': ['house_number']
	}

	data_map_text = {
		'place': ['city', 'province'],
		'country': ['country', 'country'],
		'gender': ['gender'],
		'legal_type': ['legal_type'],
		'pers_addr' : ['address', 'person_name'],
		'comp_addr': ['company_name', 'sbi_description']
	}
	data_folder = 'data_train/'

	number_of_columns = 80
	examples_per_class = 60
	total_actual = []
	total_predicted = []
	
	tmp = []
	exp_actual = []
	exp_predicted = []
	sf_main = Storage_Files(data_folder, data_map_main)
	sf_numbers = Storage_Files(data_folder, data_map_numbers)
	sf_text = Storage_Files(data_folder, data_map_text)
	sf_date_tel = Storage_Files(data_folder, ['date', 'telephone_nr'])
	sf_kvk_sbi = Storage_Files(data_folder, ['kvk_number', 'sbi_code'])
	sf_pers_addr = Storage_Files(data_folder, ['person_name', 'address', 'country'])

	sf_place = Storage_Files(data_folder, ['city', 'province', 'legal_type'])
	sf_comp_addr = Storage_Files(data_folder, ['company_name', 'sbi_description', 'country', 'legal_type'])
	ccc = Column_Classification_Config()
	ccc.add_feature('main', 'Syntax_Feature_Model', [sf_main, 1, 5000, False, True])
	# numbers
	ccc.add_feature('numbers', 'Corpus', [sf_numbers, 100, 0, False, True])
	ccc.add_feature('date_tel', 'Fingerprint', [sf_date_tel, number_of_columns, examples_per_class, False, False])
	ccc.add_feature('kvk_sbi', 'Number_Feature', [sf_kvk_sbi, 100, 0, False, False])

	# Text
	ccc.add_feature('name', 'Corpus', [sf_text, 100, 0, False, True])
	ccc.add_feature('feature_place', 'Fingerprint', [sf_place, number_of_columns, examples_per_class, False, False])
	ccc.add_feature('pers_addr', 'Fingerprint', [sf_pers_addr, number_of_columns, examples_per_class, False, False])
	ccc.add_feature('feature_comp_addr', 'Corpus', [sf_comp_addr, 100, 0, False, False])

	ccc.add_matcher('main', 'Syntax_Matcher', {'main': 'syntax'}) # main classifier
	# Numbers
	ccc.add_matcher('numbers_matcher', 'Word2Vec_Matcher', {'numbers': 'corpus'}, ('main', 'numbers'))
	ccc.add_matcher('date_tel_matcher', 'Fingerprint_Matcher', {'date_tel': 'fingerprint'}, ('main', 'date_tel'))
	ccc.add_matcher('kvk_sbi_matcher', 'Number_Matcher', {'kvk_sbi': 'number_feature'}, ('numbers_matcher', 'kvk_sbi'))

	# Text
	ccc.add_matcher('text_matcher', 'Word2Vec_Matcher', {'name': 'corpus'}, ('main', 'text'))
	ccc.add_matcher('match_place', 'Fingerprint_Matcher', {'feature_place': 'fingerprint'}, ('text_matcher', 'place'))
	ccc.add_matcher('match_pers_addr', 'Fingerprint_Matcher', {'pers_addr': 'fingerprint'}, ('text_matcher', 'pers_addr'))
	ccc.add_matcher('match_comp_addr', 'Word2Vec_Matcher', {'feature_comp_addr': 'corpus'}, ('text_matcher', 'comp_addr'))

	sm = Schema_Matcher(ccc)
	# execute_test(sm, 'data_test/', True)
	# store_file('matcher', sm)

