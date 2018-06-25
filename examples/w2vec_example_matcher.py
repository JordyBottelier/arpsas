from schema_matcher import *

data_folder = "data/"
classes = ['city', 'postcode', 'kvk_number', 'address', 'company_name', 
		'email', 'telephone_nr', 'domain_name', 'bik', 'sbi_code', 'description', 'legal_type']
sf_corpus = Storage_Files(data_folder, classes)
c = Corpus(sf_corpus, 15 , 0, True)
featuremap_word2vec = {"corpus": c}
wm = Word2Vec_Matcher(featuremap_word2vec)
wm.execute_test()