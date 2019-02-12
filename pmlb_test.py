from pmlb import fetch_data

# Returns a pandas DataFrame
adult_data = fetch_data('adult')
print(adult_data.describe())

# Returns NumPy arrays
adult_X, adult_y = fetch_data('adult', return_X_y=True, local_cache_dir='./')
print(adult_X)
print(adult_y)

# Print dataset names
from pmlb import dataset_names
print(dataset_names)

# Prints classification and regression sets
from pmlb import classification_dataset_names, regression_dataset_names

print(classification_dataset_names)
print('')
print(regression_dataset_names)

# Example comparison (starter code from Github documentation)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sb

from pmlb import fetch_data, classification_dataset_names

logit_test_scores = []
gnb_test_scores = []

count = 1
for classification_dataset in classification_dataset_names:
    if count < 25:
        X, y = fetch_data(classification_dataset, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y)
    
        logit = LogisticRegression()
        gnb = GaussianNB()
    
        logit.fit(train_X, train_y)
        gnb.fit(train_X, train_y)
    
        logit_test_scores.append(logit.score(test_X, test_y))
        gnb_test_scores.append(gnb.score(test_X, test_y))
        count += 1
    else:
        break



# With AutoSK-Learn
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

base_autoskl_scores = []
count = 1
for classification_dataset in classification_dataset_names:
    if count < 25:
        print("Auto-SKLearn, on set ", count)
        X, y = fetch_data(classification_dataset, return_X_y=True)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
    
        automl = autosklearn.classification.AutoSklearnClassifier()
        automl.fit(X_train, y_train)
        
        base_autoskl_scores.append(automl.score(test_X, test_y))
        count += 1
    else:
        break
    
sb.boxplot(data=[logit_test_scores, gnb_test_scores, base_autoskl_scores], notch=True)
plt.xticks([0, 1], ['LogisticRegression', 'GaussianNB', 'AutoML'])
plt.ylabel('Test Accuracy')
plt.savefig('comparison1.png')