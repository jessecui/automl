from pmlb import fetch_data

import warnings
warnings.filterwarnings("ignore")

# Print dataset names
from pmlb import dataset_names

# Prints classification and regression sets
from pmlb import classification_dataset_names, regression_dataset_names

import matplotlib.pyplot as plt
import seaborn as sb

# With AutoSK-Learn=
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from pmlb import fetch_data, classification_dataset_names

# To test results via different times
times = [300, 600, 1500, 1800]
time_costs = {}
for timecap in times:
    print('CURRENT TIME IS ', timecap)
    base_autoskl_scores = []
    datasets_tested = []
    
    count = 1
    for classification_dataset in classification_dataset_names:
        if count <= 5:
            datasets_tested.append(str(classification_dataset))            
            
            print("Auto-SKLearn, on set ", count)
            X, y = fetch_data(classification_dataset, return_X_y=True)
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
        
            # Autosklearn classifier with 20 min limit
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task = timecap)
            print("Current X_train has size ", str(X_train.shape))
            print("Auto-SKLearn, fitting on set ", count)
            automl.fit(X_train, y_train)
            print("Auto-SKLearn, testing on set ", count)        
            base_autoskl_scores.append(automl.score(X_test, y_test))
            print("Auto-SKLearn, finished testing on set ", count)  
            
            count += 1
        else:
            break
        
    time_costs[str(timecap)] = base_autoskl_scores
    
    print("Current time Autosklearn scores: ", str(base_autoskl_scores))
    print("Datasets tested", str(datasets_tested))


    time_data = list(time_costs.values())
    time_labels = list(time_costs.keys())
    ticks = list(range(0,len(time_costs)))
    sb.boxplot(data=time_data, notch=True)
    plt.xticks(ticks, time_data)
    plt.ylabel('Test Accuracy')
    plt.savefig('comparison_times.png')
    plt.close()