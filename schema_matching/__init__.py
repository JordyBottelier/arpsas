from os.path import dirname, basename, isfile
import glob
import re
import inspect
import sys
modules = glob.glob(dirname(__file__)+"/*.py")
from .misc_func import *
__all__ = []
__all__.append('print_dict')
__all__.append('store_file')
__all__.append('read_file')
__all__.append('store_matcher')
__all__.append('read_matcher')
__all__.append('print_list')
for f in modules:
	if isfile(f) and not f.endswith('__init__.py'):
		filename = basename(f)[:-3]
		__all__.append(basename(f)[:-3])
		# Extract the class itself and add it to the imports
		if "_base" not in filename:
			texfile=open(f, "r")
			for line in texfile:
				if re.match("class .*\:", line):
					class_name = re.sub('class', '', line)
					class_name = re.sub(':', '', class_name)
					class_name = re.sub('\(.*\)', '', class_name)
					class_name = class_name.strip()
					command = "from ." + filename + " import " + class_name
					exec(command)
					__all__.append(class_name)

# also import all the classes from the features and matchers so they can be tested
from .feature_classes import all_classes
for clazz in all_classes:
	command = "from .feature_classes import " + clazz
	exec(command)
	__all__.append(clazz)

# also import all the classes from the features and matchers so they can be tested
from .column_classifiers import all_classes
for clazz in all_classes:
	command = "from .column_classifiers import " + clazz
	exec(command)
	__all__.append(clazz)
