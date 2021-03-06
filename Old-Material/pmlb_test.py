from pmlb import fetch_data

import warnings
warnings.filterwarnings("ignore")

# Print dataset names
from pmlb import dataset_names

# Prints classification and regression sets
from pmlb import classification_dataset_names, regression_dataset_names

# Example comparison (starter code from Github documentation)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sb

# With AutoSK-Learn
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from pmlb import fetch_data, classification_dataset_names

logit_test_scores = []
gnb_test_scores = []
base_autoskl_scores = []
datasets_tested = []

count = 1
for classification_dataset in classification_dataset_names:
    if count <= 2:
        datasets_tested.append(str(classification_dataset))
        X, y = fetch_data(classification_dataset, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y)
    
        logit = LogisticRegression()
        gnb = GaussianNB()
    
        logit.fit(train_X, train_y)
        gnb.fit(train_X, train_y)
    
        logit_test_scores.append(logit.score(test_X, test_y))
        gnb_test_scores.append(gnb.score(test_X, test_y))
        
        
        print("Auto-SKLearn, on set ", count)
        X, y = fetch_data(classification_dataset, return_X_y=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    
        # Autosklearn classifier with 20 min limit
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task = 300)
        print("Current X_train has size ", str(X_train.shape))
        print("Auto-SKLearn, fitting on set ", count)
        automl.fit(X_train, y_train)
        print("Auto-SKLearn, testing on set ", count)        
        base_autoskl_scores.append(automl.score(X_test, y_test))
        print("Auto-SKLearn, finished testing on set ", count)  
        
        count += 1
    else:
        break


print("Autosklearn scores: ", str(base_autoskl_scores))
print("Logit scores: ", str(logit_test_scores))
print("GNB scores: ", str(gnb_test_scores))
print("Datasets tested", str(datasets_tested))
    
sb.boxplot(data=[logit_test_scores, gnb_test_scores, base_autoskl_scores], notch=True)
plt.xticks([0, 1, 2], ['LogisticRegression', 'GaussianNB', 'AutoML'])
plt.ylabel('Test Accuracy')
plt.savefig('comparison1b.png')