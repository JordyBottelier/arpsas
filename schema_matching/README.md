# Learning-Based Schema Matching Algorithm Test Framework

This repo contains the information that is needed to implement and use the schema matching framework that has been created by Jordy Bottelier. 
For questions or suggestions please feel free to email me at bottelierjordy@gmail.com. 

This framework allows for a user to create and test their own learning or decision based schema matching algorithm. 
The complete frameworks works in 5 steps, at any point you can try and test your own implementations since they are imported automatically. 

The steps are as follows (for instructions on the steps please click the links):
1. [Collection of data](feature_classes/#data-collector)
2. [Creating features](feature_classes/#features)
3. [Creating a column classifier](column_classifiers/)
4. [Creating a pipeline of column classifiers](#column-classification-pipelines)
5. [Using the pipeline to match schemas](#using-the-pipeline)

## Importing framework

## Column Classification Pipelines
If you have created feature classes and matcher classes that utilize these features, you are now ready to create a column classification config /pipeline. You can use the Column_Classification_Config object to build a tree structure and add submatchers to matchers. 

### Step 1: Creating and Adding Features
The first thing you need to do to when building a new pipeline is add the features. You can use the add_feature method for this:
```python
data_folder = "data/"

data_map1 = {
	'place': ['address', 'postcode'],
	'numbers': ['sbi_code', 'kvk_number', 'bik'],
	'name': ['company_name', 'city', 'legal_type'],
	'contact': ['telephone_nr', 'email']
}
sf_fingerprint_main = Storage_Files(data_folder, data_map1)

ccc = Column_Classification_Config()
ccc.add_feature("feature_main", "Fingerprint", [sf_fingerprint_main, 50, 0, False, True])
```
The first argument is a name you give to the feature which you can use later. Then you specify what kind of feature class you would like to use. In this case we use the build-in Fingerprint feature class (if you add classes into the feature_classes folder they are automatically imported and can be used as well). The third argument is a list of arguments used for the initialization of the feature. 

You can keep adding features to the object but remember that each one must contain a unique feature name:
```python
sf_fingerprint_place = Storage_Files(data_folder, ['address', 'postcode'])
sf_fingerprint_numbers = Storage_Files(data_folder, ['sbi_code', 'kvk_number', 'bik'])
ccc.add_feature("feature_place", "Fingerprint", [sf_fingerprint_place, 50, 0, False, False])
ccc.add_feature("feature_numbers", "Fingerprint", [sf_fingerprint_numbers, 50, 0, False, False])
``` 
If you do not understand how the Storage_Files object and feature objects work then please go back to step 1. 

### Step 2: Creating and Adding Column Classifiers
Once we have added features, we can now add matchers. Internally the Column_Classification_Config will create a tree-structure in which the matchers are stored. A matcher can have multiple submatchers which are called upon pre-defined classes. We can add matchers to the tree as follows:

```python
# main classifier, no final optional argument since this is the base matcher
ccc.add_matcher('fingerprint_matcher', 'Fingerprint_Matcher', {"feature_main": "fingerprint"}) 

# sub classifiers for main classifier
ccc.add_matcher('match_place', 'Fingerprint_Matcher', {"feature_place": "fingerprint"}, ('fingerprint_matcher', 'place'))
ccc.add_matcher('match_numbers', 'Fingerprint_Matcher', {"feature_numbers": "fingerprint"}, ('fingerprint_matcher', 'numbers'))
```

* The first argument is the name of the matcher (must be unique in the pipeline). 
* The second argument is the type of matcher you want to utilize (if you add classes into the column_classifiers folder they are automatically imported and can be used as well). 
* Thirdly we specify the featuremap, the keys are the names of the features you defined in the previous step, and the values are the keys you used in the column classification step (step 3).
* The final argument is used when you want to add sub-matchers to a matcher. The first argument is the parent matcher (which will be looked up in the tree), and the second argument is the classname upon which you'd like to invoke this sub classifier. 

In this example, the main classifier will predict one of the four classes: 'name', 'number', 'place' or 'contact'. When the 'place' class is predicted, our newly created matcher 'match_place' is called to further classify the data into the classes 'address' or 'postcode'. If no submatcher is called the original result from the classifier is returned. 

You can use this structure to keep expanding your column classifier until it perfectly satisfies your needs. The Match_Tree which is created by the Column_Classification_Object is then used by the Schema_Matcher to classify columns. 

### Step 3: Using The Pipeline
If you load the Column_Classification_Object into a Schema_Matcher object you can already use your newly build pipeline to classify columns or instances. Columns are classified by walking the tree, and calling the matchers and submatchers classification methods upon the columns:
``` 
sm = Schema_Matcher(ccc)
sm.classify_column([['EINDHOVEN', 'ENSCHEDE', 'EINDHOVEN', "'S-HERTOGENBOSCH", 'EINDHOVEN', 'EINDHOVEN', 'NIJMEGEN']])
```
Note that the arguments are passed to the classification methods as a list. 

You can currently also invoke the execute_test methods on the matchers, but be aware, this might break if you do not use enough classes, and if you have implemented your own matchers which do not have this method. 


