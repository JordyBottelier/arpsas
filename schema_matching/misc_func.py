import pickle
import pprint
import sys
import collections

def store_file(filename, data):
	pickle.dump(data, open(filename, "wb"))

def store_matcher(filename, data):
	pickle.dump(data, open(('trained_matchers/' + filename), "wb"))

def read_matcher(filename):
	return pickle.load(open(('trained_matchers/' + filename), "rb" ), encoding='latin1')

def read_file(filename):
	return pickle.load(open(filename, "rb" ), encoding='latin1')

def store_file_string(filename, data):
	f = open(filename, "wb")
	f.write(str(data))

def print_dict(data):
	pp = pprint.PrettyPrinter(indent=4)
	pp.pprint(data)

def print_list(data):
	for n in data:
		print(n)

def throw_error(errorstring):
	print(errorstring)
	sys.exit()

def precision(actual, predicted):
	tp = 0
	fn = 0
	for i in range(0, len(actual)):
		predic = predicted[i]
		act = actual[i]
		if predic != "unknown":
			if predic == act:
				tp += 1
			else:
				fn += 1
	return tp / float(tp + fn)

def recall(actual, predicted):
	tp = 0
	fp = 0
	for i in range(0, len(actual)):
		predic = predicted[i]
		act = actual[i]
		if act != "unknown":
			if predic == act:
				tp += 1
			else:
				fp += 1
	return tp / float(tp + fp)


def f_measure(actual, predicted):
	prec = precision(actual, predicted)
	rec = recall(actual, predicted)
	return 2 * (prec * rec) / (prec + rec)