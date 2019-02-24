
# coding: utf-8
# Author: Jesse Cui
# Program to run Auto-SkLearn Benchmarks on PMLB Datasets
# Input: Dataset range, Dataset type (classification or regression), training times
# Output: CSV file of performance results

# Surpress warnings
import warnings
warnings.filterwarnings("ignore")

# Import libraries
from pmlb import dataset_names, classification_dataset_names, regression_dataset_names, fetch_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

# Import SK-learn and AutoSK-Learn
import autosklearn.classification
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Argument Parsers for the following:
# Min dataset number (default is 1)
# Max dataset number (default is 166 for classifcation, 120 for regression)
# Mutually exclusive argument for classification vs regression dataset
# List of number of seconds to train datasets on (default is 3600)

import argparse
parser = argparse.ArgumentParser(description='Run Auto-SkLearn on PMLB datasets')

parser.add_argument('-min', '--minset', type=int, metavar='', required=False, default=1, help = 'Min dataset number (default 1)')
parser.add_argument('-max', '--maxset', type=int, metavar='', required=False, default=166, help = '# Max dataset number (default is 166 for classifcation, 120 for regression)')
parser.add_argument('-t', '--times', type=int, nargs='+', metavar='', required=False, default=3600, help = 'List of number of seconds to train datasets on (default is 3600)')

class_group = parser.add_mutually_exclusive_group()
class_group.add_argument('-r', '--regre_sets', action='store_true', help='Benchmark on regression sets')
class_group.add_argument('-c', '--class_sets', action='store_true', help='Benchmark on classification sets (default)')

args = parser.parse_args()

# Assign variables based on arguments
minset = args.minset
maxset = args.maxset
times = args.times
regre_sets = args.regre_sets
class_sets = args.class_sets

# Set classification sets to default if no class was selected
if not regre_sets and not class_sets:
    class_sets = True

# Rescale dataset max number to be within boundaries
if minset < 1:
    minset = 1
    print('Minset provided is less than 1, changed to 1.')
if class_sets and maxset > 166:
    maxset = 166
    print('Maxset provided is greater than 166, changed to 166.')
if regre_sets and maxset > 120:                
    maxset = 120
    print('Maxset provided is greater than 120, changed to 120.')    
       
# Create a dictionary of the number of features, instances, and classes per classification dataset
# Potentially look into including number of binary, integer, and float features in the future
datasets = []
dataset_props = {}

if class_sets:
    dataset_names = classification_dataset_names[minset-1: maxset] 
if regre_sets:
    dataset_names = regression_dataset_names[minset-1: maxset]

dataset_number = minset;
for dataset in dataset_names:
    X, y = fetch_data(dataset, return_X_y=True)
    num_instances, num_features =  X.shape
    num_classes = (np.unique(y)).size if class_sets else -1
    dataset_props[dataset] = (num_instances, num_features, num_classes, dataset_number)
    dataset_number += 1

# Add to this dataframe the performance results of the datasets that we query on
df_rows_list = []
for time_cap in times:
    print('CURRENT TIME IS ', time_cap)
        
    for dataset in dataset_names:
        curr_dataset_results = {}
        print("Auto-SKLearn, on dataset ", dataset, " | Number: ", str(dataset_props[dataset][3]), "max of ", str(maxset))
        print("Properties: ")
        print(str(dataset_props[dataset]))
        
        # Split the data to training and test sets
        X, y = fetch_data(dataset, return_X_y=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

        # Run the classifier
        automl = 0;
        
        if class_sets:
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task = time_cap)
        if regre_sets:
            automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task = time_cap)                       
            
        print("Auto-SKLearn, fitting")
        automl.fit(X_train, y_train)
        print("Auto-SKLearn, testing")        
        current_score = automl.score(X_test, y_test)                            
        print("Auto-SKLearn, finished testing on set ", str(dataset_props[dataset][3]))
        print("Current time Autosklearn score: ", str(current_score))
              
        # Store the result in a dictionary
        curr_dataset_results['name'] = dataset
        curr_dataset_results['number'] = dataset_props[dataset][3]
        curr_dataset_results['num_instances'] = dataset_props[dataset][0]
        curr_dataset_results['num_features'] = dataset_props[dataset][1]
        curr_dataset_results['num_classes'] = dataset_props[dataset][2]
        curr_dataset_results['time_cap'] = time_cap
        curr_dataset_results['score'] = current_score
              
        # Append current dictionary to a list of dictionary
        df_rows_list.append(curr_dataset_results)                   
        
# Create a Pandas Dataframe with the results
autosklearn_df = pd.DataFrame(df_rows_list)

# Save results into a CSV
set_type_string = 'c' if class_sets else 'r'

times_string = ''

for i in range(len(times)):
    times_string += str(times[i])
    if i != len(times) - 1:
        times_string += '_'

file_name = set_type_string + '_' + str(minset) + '_' + str(maxset) + '_' + 'times' + '_' + times_string + '.csv'
print('saving to ', file_name)

autosklearn_df.to_csv(file_name, sep='\t')

