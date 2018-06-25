from schema_matcher import *
data_folder = "data/"
data_map = {
		'place': ['address', 'postcode'],
		'numbers': ['sbi_code', 'kvk_number', 'bik'],
		'name': ['company_name', 'city', 'legal_type'],
		'contact': ['telephone_nr', 'email']
	}

sf_sfm = Storage_Files(data_folder, data_map)
sfm = Syntax_Feature_Model(sf_sfm, 1, 1000, False, use_map=True)
featuremap_sm = {"syntax": sfm}
sm = Syntax_Matcher(featuremap_sm)
sm.execute_test()
sm_col = ["Wilhelminalaan 37", "Eikenlaan 28", "EINDHOVEN", "1161TW"]
print(sm.classify_column(sm_col))
print(sm.classify_instance("Jordy@bottelier.org"))