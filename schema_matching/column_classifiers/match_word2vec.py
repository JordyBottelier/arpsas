from .match_base import Matcher
import numpy as np
from .tester import Tester
from gensim.models import Word2Vec


class Word2Vec_Matcher(Matcher, Tester):
	"""
		Create a classifier based on the word2ved natural language model. 
		In order to make this work we treat every incoming column as a corpus.
		The columns are already split and prepared by the feature class. 

		For training/predicting we calculate the mean score of the corpus in a column and use this
		as a feature vector for the classifier. 
	"""
	def __init__(self, featuremap):
		Matcher.__init__(self, featuremap)
		self.train()

	def train(self):
		corpus = self.featuremap['corpus']
		corpus, targetpoints = corpus.get_features_targets()
		self.train_manual(corpus, targetpoints)

	def train_manual(self, corpus, targetpoints):
		"""
			Train the classifier by first computing vectors from the w2vec model.
			Per column a word2vec model is created, and the datapoints are the mean scores of the corpus in the word2vec model. 
			A classifier is then trained upon these mean datapoints
		"""
		self.model = Word2Vec(corpus, min_count=1, size=200)
		self.w2v = dict(zip(self.model.wv.index2word, self.model.wv.syn0))
		self.dim = len(list(self.w2v.values())[0])
		datapoints = self.mean_scores(corpus)
		self.create_oneclass_dict(datapoints, targetpoints)
		self.clf.fit(datapoints, targetpoints)

	def classify_instance(self, entry):
		corpus = self.featuremap['corpus']
		col_tokenized = corpus.extract_features_column([entry])
		datapoints = self.prediction_score(col_tokenized)
		return self.clf.predict(datapoints.reshape(1, -1))

	def classify_instance_proba(self, entry):
		corpus = self.featuremap['corpus']
		col_tokenized = corpus.extract_features_column([entry])
		datapoints = self.prediction_score(col_tokenized)
		return self.clf.predict_proba(datapoints.reshape(1, -1))

	def classify_column(self, column, detect_outlier=False):
		corpus = self.featuremap['corpus']
		col_tokenized = corpus.extract_features_column(column)
		datapoints = self.prediction_score(col_tokenized)
		prediction = self.clf.predict(datapoints.reshape(1, -1))
		if detect_outlier:
			outlier = self.outlier_detector_dict[prediction[0]].predict(datapoints.reshape(1, -1))
			return self.clf.predict(datapoints.reshape(1, -1)), outlier
		return prediction

	def classify_column_proba(self, column, detect_outlier=False):
		corpus = self.featuremap['corpus']
		col_tokenized = corpus.extract_features_column(column)
		datapoints = self.prediction_score(col_tokenized)
		prediction = self.clf.predict(datapoints.reshape(1, -1))
		if detect_outlier:
			outlier = self.outlier_detector_dict[prediction[0]].predict_proba(datapoints.reshape(1, -1))
			return self.clf.predict_proba(datapoints.reshape(1, -1)), outlier
		return self.clf.predict_proba(datapoints.reshape(1, -1))

	def prediction_score(self, column_tokens):
		"""
			Get the scoring vector of the column tokens based on the model. 
			Use the average if the total number of entries is bigger than 0.
		"""
		score = np.zeros(self.dim)
		num_entries = 0
		for token in column_tokens:
			if token in self.w2v:
				num_entries += 1
				score += self.w2v[token]
		if num_entries > 1:
			score = score / float(num_entries)
		return score

	def mean_scores(self, corpus):
		""" 
			We use the model to get the mean score of the entire corpus/column, 
			and use this to train the classifier. 
		"""
		scores = []
		for column in corpus:
			total = np.zeros(self.dim)
			for word in column:
				total += self.w2v[word]
			scores.append(total / float(len(column)))
		return np.array(scores)

	def classify_prepared_instance(self, entry):
		target_score = self.prediction_score(entry)
		return self.clf.predict(target_score.reshape(1, -1))[0]

	def execute_test(self, num_tests=5, learnset_ratio=0.7):
		corpus = self.featuremap['corpus']
		corpus, targetpoints = corpus.get_features_targets()
		train_arguments = [corpus, targetpoints, learnset_ratio]

		return self.k_fold_test_classifier(self.create_datasets, train_arguments, 
			self.train_manual, self.classify_prepared_instance, num_tests)