# Column Classifiers (Matchers)
A column classifier (also frequently called matcher) uses the datapoints defined by the features, and applies these to a given machine learning or matching algorithm/strategy.

## Behaviour
Multiple features can be utilized by a single matcher, you can pass all the features you want by using a featuremap. This is simply a dictionary in which the key is the name of your feature, and the value is the feature object you already initialized. Each matcher has a classification strategy or machine learning element (Already present in the base class). 

The following behaviour is expected from each matcher:
* (optional but recommended) It is trained upon initialization by utilizing the data from the given feature objects. 
* (optional but recommended) It has a train() method which is called upon initialization, and a classifier is trained. If you do not use a classifier (but more of a pre-build set of rules), just create the method and pass it, and dont call it upon initialization. 
* (required) There is a working classify_column method, which classifies a column and then returns the class or list of classes.
* (required) There is a working classify_instance method, which classifies an entry and then returns the class or list of class.
* (optional but recommended) There is a working classify_column_proba method, which classifies a column and then returns the class probability distribution.
* (optional but recommended) There is a working classify_instance_proba method, which classifies an entry and then returns the class probability distribution.
* (optional but recommended) There is a working execute_tests method. This method uses the k_fold_test of the Tester class which is explained later. 

Most of the optional methods are useful for testing and using a machine learning based classifier. 

Once you have created a matcher or multiple matchers you are ready to create a column classification pipeline. 
You can also use matchers individually in your own implementation outside of the pipeline like such:
```python
	data_folder = "data/"
	classes = ['city', 'postcode', 'kvk_number']
	sf_fingerprint = Storage_Files(data_folder, classes)
	# We want to simulate 30 columns per data class with 100 entries per column
	f = Fingerprint(sf_fingerprint, 30, 100, unique=False, use_map=False)
	featuremap = {"fingerprint": f}
	fm = Fingerprint_Matcher(featuremap)

	# Some tests
	tmp = ['0204975850', '0611255632']
	print(fm.classify_column(tmp))
	print(fm.classify_instance_proba("0626338823"))
	fm.execute_test()
```
Here we use the build in Fingerprint feature, and Fingerprint_Matcher to classify some data.

## Testing
Testing a matcher can be done using the highly versatile test class. The testclass has the following methods:

```python
def k_fold_test_classifier(self, dataset_method, dataset_args, training_method, prediction_method, num_tests=5):
		""" 
			Runs k random tests.
			You have to specify the method that is needed to provide the classifier with the learn and testset datapoints.
			Pass the arguments as a list of ordered arguments needed by the function

			Make sure any method that gets a dataset to run in this test returns in the following order:
			learnset_datapoints, learnset_targetpoints, testset_datapoints, testset_targetpoints.

			the training method needs to train itself depending on the learnset_datapoints and learnset_targetpoints.

			The prediction method needs to be able to work with instances of the testset_datapoints, and needs to return 
			entities present in testset_targetpoints

			These points need to be pre-processed.
		"""

def get_results_single_test(self, learnset_datapoints, learnset_targetpoints, testset_datapoints, \
			testset_targetpoints, training_method, prediction_method):
		
		""" Train the model and test the accuracy """

def create_datasets(self, datapoints, targetpoints, learnset_ratio):
		""" Split the dataset in a test and learn set 
			kwargs should be: datapoints, targetpoints, learnset_ratio
		"""

```



