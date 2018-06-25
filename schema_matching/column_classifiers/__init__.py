from os.path import dirname, basename, isfile
import glob
import re
modules = glob.glob(dirname(__file__)+"/*.py")
all_classes = []

__all__ = []
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
					all_classes.append(class_name)