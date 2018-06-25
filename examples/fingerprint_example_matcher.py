from schema_matcher import *

data_folder = "data/"
classes = ['city', 'postcode', 'kvk_number', 'address', 'company_name', 
		'email', 'telephone_nr', 'domain_name', 'bik', 'sbi_code', 'description', 'legal_type']

sf_fingerprint = Storage_Files(data_folder, classes)
f = Fingerprint(sf_fingerprint, 30, 100, unique=False, use_map=False)
featuremap = {"fingerprint": f}
fm = Fingerprint_Matcher(featuremap)
tmp = ['0204977050']
print(fm.classify_column(tmp))
print(fm.classify_instance_proba("0626338823"))
fm.execute_test()