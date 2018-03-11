import random
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold
from feature_engineering import *
from classifier_testing import *

path_to_data = "../data/"
training_features = np.load(path_to_data + 'training_features100.npy')
labels_array = np.load(path_to_data + 'labels_array100.npy')

best_score = 0
lgb_params = {}
lgb_params['n_estimators'] = 1100
lgb_params['max_depth'] = 4
lgb_params['learning_rate'] = 0.02
#lgb_params['feature_fraction'] = 0.9
#lgb_params['bagging_freq'] = 1
#lgb_params['random_state'] = 0  # Define features to be used
parameters_names = ['n_estimators', 'max_depth', 'learning_rate', 'feature_fraction', 'bagging_freq', 'random_state']
best_values = [40,2,0.1,1,1,0]
best_score = 0
options_list = [[40,75,150,300,600, 800,1200],
           [2,3,4,6,8,10],
           [0.1,0.05, 0.01, 0.005,0.001]]
for options_index in range(len(options_list)):
    for value_index in range(len(options_list[options_index])):
        lgb_params[parameters_names[options_index]] = options_list[options_index][value_index]
        classifier = LGBMClassifier(**lgb_params)
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        validation_scores = []
        cv_index = 0
        for train_index, validation_index in kf.split(training_features, labels_array):
            print("Cross-validation, Fold %d" % (cv_index + 1))
            # Split data into training and testing set
            training_features_cv = training_features[train_index, :]
            validating_features = training_features[validation_index, :]
            training_labels = labels_array[train_index]
            validating_labels = labels_array[validation_index]
            classifier = classifier.fit(training_features_cv, training_labels)
            # Test the model
            validation_pred = classifier.predict(validating_features)
            score = f1_score(validating_labels, validation_pred)
            print "score: ", score
            validation_scores.append(score)
            cv_index += 1
        mean_score = sum(validation_scores) / 5
        print "mean score: ", mean_score
        if(mean_score > best_score):
            best_score = mean_score
            best_values[options_index] = options_list[options_index][value_index]
print best_values