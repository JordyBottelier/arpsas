# Data Collection and Features
The first steps in the entire pipeline are to gather data and calculate datapoints (create the features). 
Here you can read how you can use the framework to gather data or insert it manually. Then you can read on how feature classes can
be created and used and what the conventions are. 

## Data Collector
The Data_Collector collects data from specified folders. Using the Data_Collector is optional, you can also pre-process the data yourself and pass it as a dictionary to a feature. 

### Prequisites
The Data_Collector requires that you have your data per class in serperate folders. The data can consist of multiple files. Inside a datafile should be a readable python list. All files in the folder are read, the data is mushed together, and used as entry data for the classifiers. It is recommended that you give the data folders the same name as the classes they are meant to represent. 

To get to the data the Data_Collector needs a Storage_Files object. A Storage_Files object contains a path to the folder in which the data/class folders lie. If you want the Data_Collector to read data from multiple folders and mush it into a single column/class you can use the data_map option. 
A Storage_Files object can be defined as follows:

```python
data_folder = "data/"
data_map = {
		'place': ['address', 'postcode'],
		'numbers': ['sbi_code', 'kvk_number', 'bik'],
		'name': ['company_name', 'city', 'legal_type'],
		'contact': ['telephone_nr', 'email']
	}

storage_files = Storage_Files(data_folder, data_map)
```

In this example, the Data_Collector would look in the 'data/' folder, and create columns with the classlabels: 'place', 'numbers', 'name', and 'contact'. These are the classes that are now present in the dataset. The list items per class (in the example) are the folders in which the data for the collector can be found in files. So all the data in the data/address/ folder will be mushed together with the data in the data/postcode/ folder.  

You can also use a simple list to create a Storage_Files object:
```python
data_folder = "data/"
classes = ['city', 'postcode', 'kvk_number']

storage_files = Storage_Files(data_folder, classes)
```
The Data_Collector will now look in the folder data/city and read all the files, and use this as data for the class 'city'.

### Behaviour
You can use the Data_Collector to split your columns into serperate chunks. For example, if you have collected a column with 100 entries, you can ask the Data_Collector to split the column into serveral chunks with a specified number of entries per chunk. You can use this to simulate having 10 columns.

The data collector takes the following arguments:
```python
def __init__(self, sf=None, num_columns=1, examples_per_column=0, unique=False, use_map=False):
```
* sf: The previously discussed Storage_Files object. 
* num_columns: The number of columns you want to use per class. If you have 10 classes, and specify that every class will have 4 columns, you will have a total of 40 columns in your entire dataset. 
* examples_per_column: You can limit the amount of data you want per simulated column. This is useful if you want to test with a smaller dataset, or if you want the program to be faster. If you leave the value to 0 the Data_Collector will randomly and evenly divide the data over the specified number of columns. 
* unique: If set to true, the data per column will consist of all unique values.
* use_map: Set to True if you used a data_map when creating the storage files object.

The Data_Collector creates the dictionary with columns upon initialization and stores the data in its own object. Use the getter to receive the data.

Remember that for all the pre-defined features you don't necessarily have to use the Data_Collector, you can also create a dictionary with the classes yourself, as long as you use the following format:
```python
{
	class1: [[col_data1], [col_data2]],
	class2: [[col_data3], [col_data4]]

}

```

## Features
You can create your own feature class, which collects features per column or per entry. In short, a feature collects datapoints for a column, and for training purposes stores these datapoints along with their target class. Both are stored in a list, but more on that in the behaviour section. 

### Behaviour
Every feature that you define needs to have the following behaviour:
* Features have to create their own datapoints (which are later used by the matchers) upon initialization.
* The datapoints per column or entry should be stored in a list called features (already present in the Feature_Base class).
* Per datapoint you need to also add the target (class) to the list of targets. 
* The features and datapoints need to be returned by a method called 'get_features_targets' (also standard in Feature_Base), this is later used by the matchers.
* Every feature should have a 'extract_features_column' method which extracts and returns datapoints for a given column, this is later used by the matchers for classification.

A feature usually has the following arguments in the init:
```python
def __init__(self, sf=None, num_columns=10, examples_per_column=0, unique=False, use_map=False)
```
All of these arguments are used by the data collector. The build in features can also work with a dictionary (as was specified in the previous section), just pass it as if it was the Storage_Files object, so: sf=my_dict.

You can use your own implementation to collect data and build features but if you want the pre-build matchers and features to be compatible with your own implementation that you should stick to the conventions (at least the ones defined in the list).

Once you have defined features (and created your datapoints and target classes in the process), you are ready to build a matcher.

