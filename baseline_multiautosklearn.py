import numpy as np
import sklearn.model_selection
from sklearn import datasets
import sklearn.metrics
import shutil
import autosklearn.classification
import autosklearn.regression
import pickle
import sys

import warnings
warnings.filterwarnings("ignore")

# Inputs: X, y, regression or classification
class base_multi_automl:
    def __init__(self, model_type):
        # Classification or regression indicator
        self.model_type = model_type
        # List of auto-sklearn models for each block
        self.block_models = 0
        # Final model used on the block predictions to aggregate their predictions
        self.final_model = 0        
        # List of start indices for each group in the organized 
        
    # Method to create a list of groups given a dataset and vector of each feature's assigned group
    @staticmethod
    def create_groups(X, groups):    
        total_group_num = np.size(np.unique(groups))    
        group_counts = [0] * total_group_num
        m, d = np.shape(X)

        # Create a list of the groups
        X_groups = [np.empty(1) for i in range(total_group_num)]
        for group_num in range(total_group_num):
            print('Group Num is {}.'.format(group_num))
            for feature in range(d):        
                if group_num == groups[feature]:
                    group_counts[group_num] += 1
                    if group_counts[group_num] == 1:
                        X_groups[group_num] = np.asmatrix(X[:, feature]).T                
                    else:
                        X_groups[group_num] = np.hstack((X_groups[group_num], np.asmatrix(X[:, feature]).T))                 
        return X_groups, group_counts
                        
    # Method to create a data table for model predictions of each classifier
    @staticmethod
    def getBlockPredictions(X_groups, block_models):
        m = X_groups[0].shape[0]
        total_group_num = len(X_groups)
        block_predictions = np.empty((m, total_group_num))
        for group_num in range(total_group_num):
            print('GROUP NUM ', group_num)
            group_automl_model = pickle.loads(block_models[group_num])

            block = X_groups[group_num]
            block_predictions[:, group_num] = group_automl_model.predict(block)
        return block_predictions        
    
    # X and y are NP matrices, assumes groups is an np vector with indicators starting from 0
    # automl_time_cap and automl_model_run_time_limit are used in the auto-sklearn subroutine
    def fit(self, X, y, groups, automl_time_cap = 600, automl_model_run_time_limit = 60):
        # Retrieve the total number of groups
        m, d = np.shape(X)
        total_group_num = np.size(np.unique(groups))
        
        # Retrieve the groups of data and the feature counts for each group
        print('CREATING GROUPS')
        X_groups, group_counts = base_multi_automl.create_groups(X, groups)
        
        # Train each group on AutoML and save the model
        print('TRAINING AUTOML ON GROUPS')
        self.block_models = [0] * total_group_num
        for group_num in range(total_group_num):
            print('GROUP NUM ', group_num)
            block_data = X_groups[group_num]
            automl_model = 0
            if self.model_type == 'classification':
                automl_model = autosklearn.classification.AutoSklearnClassifier(
                            time_left_for_this_task = automl_time_cap,
                            per_run_time_limit = automl_model_run_time_limit)
            automl_model.fit(block_data, y)
            self.block_models[group_num] = pickle.dumps(automl_model)
        
        print('RETRIEVING PREDICTIONS FOR EACH GROUP')
        block_predictions = base_multi_automl.getBlockPredictions(X_groups, self.block_models)
        
        # Create one final ensemble method to train on the block predictions for one final prediction
        if self.model_type == 'classification':
            from sklearn.linear_model import LogisticRegression
            self.final_model = LogisticRegression()

        self.final_model.fit(block_predictions, y)  
        
    def predict(self, X, groups, block_presence):
        # Groups should be of the same structure as before
        # Block presence indicates which blocks are missing or not                

        # Retrive the sorted datasets for training and testing datasets
        print('CREATING GROUPS')
        X_groups, group_counts = base_multi_automl.create_groups(X, groups)
        
        print('RETRIEVING PREDICTIONS FOR EACH GROUP')
        block_predictions = base_multi_automl.getBlockPredictions(X_groups, self.block_models)
        return self.final_model.predict(block_predictions)        
     

        
        
        
        