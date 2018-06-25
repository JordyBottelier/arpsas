from schema_matching import *

if __name__ == '__main__':
	data_folder = 'data_train/'
	classes = ['city', 'postcode', 'kvk_number', 'address', 'company_name', 
			'email', 'telephone_nr', 'domain_name', 'bik', 'description', 'legal_type']

	classes_test = ['address', 'postcode', 'kvk_number', 'bik', 'company_name', 'city', 'legal_type', 'telephone_nr', 'email', 'sbi_code', 'domain_name']
	data_map = {
		'numbers': ['kvk_number', 'postcode', 'telephone_nr'],
		'name': ['company_name', 'city', 'legal_type', 'bik'],
		'email': ['email'],
		'address': ['address'],
		'domain_name': ['domain_name']
	}

	data_map_numbers = {
		'kvk_sbi': ['kvk_number', 'sbi_code'],
		'postcode': ['postcode'], 
		'telephone_nr': ['telephone_nr'],
		'kvk_number' : ['kvk_number']
	}

	sf_kvk_sbi = Storage_Files(data_folder, ['kvk_number', 'sbi_code'])
	nf = Number_Feature(sf_kvk_sbi, 50, 100, unique=False, use_map=False)
	featuremap = {"number_feature": nf}
	fm = Number_Matcher(featuremap)
	tmp = ['01000638', '01002252', '01002411', '01011684', '01012837', '01016752', '01017276', '01021095', '01026596', '01028053']
	print(fm.classify_column(tmp))
	# print(fm.classify_instance_proba("0626338823"))
	print(fm.execute_test())