import random
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel
import nltk
import csv
import sys
from feature_engineering import *
from classifier_testing import *
from read_data import *
from graph_creation import *
from sklearn.ensemble import RandomForestClassifier


# ---Parameters--- #
submission_mode = True
testing_mode = False
quick_eval_mode = True
classifier_tuning_mode = True
probabilistic_mode = True
submission_name = "prob_random_forest_tuned_0.1_g1and2_wmd_auth"
TRAINING_SUBSAMPLING = 0.1
LOCAL_TEST_SUBSAMPLING = 0.01
seed = 1337

print "training subsample: ", TRAINING_SUBSAMPLING
print "testing mode: ", testing_mode
if testing_mode:
    print "testing subsample: ", LOCAL_TEST_SUBSAMPLING
print "submission mode: ", submission_mode
if submission_mode:
    print "submitting with name: ", submission_name

# ---First Initializations--- #
random.seed(seed)  # to be able to reproduce results
path_to_predictions = "../predictions/"
nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

###############################
# beating the random baseline #
###############################
# the following script gets an F1 score of approximately 0.66

# ---Read Data--- #
testing_set, training_set, node_info = read_data()
# the columns of the node_info data frame are:
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes
IDs = [element[0] for element in node_info]

# ---Compute TFIDF vector of each paper--- #
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

# ---Create graph--- #
g = create_graph(training_set, IDs)

# ---Training--- #
print "Training"
# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*TRAINING_SUBSAMPLING)))
training_set_reduced = [training_set[i] for i in to_keep]
# create training features
if quick_eval_mode:
    print "Loading pre-trained features"
    training_features = np.load('./data/training_features10.npy')
    labels_array = np.load('./data/labels_array10.npy')
else:
    training_features = feature_engineering(training_set_reduced, IDs, node_info, stemmer, stpwds, g)
    np.save('./data/training_features.npy', training_features)
    # convert labels into integers then into column array
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    np.save('./data/labels_array.npy', labels_array)
    print "Features calculated"

# initialize classifier(s)
if classifier_tuning_mode:
    #tune_rf(train_x=training_features, train_y=labels_array,seed=seed)
    tune_SVC(train_x=training_features, train_y=labels_array)
    sys.exit(0)
else:
    clfs = []
    clfs.append(RandomForestClassifier(n_estimators=45, max_depth=25, min_samples_leaf=2, random_state=seed))
    clfs.append(svm.SVC(C=5, gamma=0.015))
    clfs.append(LGBMClassifier())

# train model with features and labels
for classifier in clfs:
    classifier.fit(training_features, labels_array)
print "Training done"

# ---Test--- #
if testing_mode:
    for classifier in clfs:
        print "\n\nClassifier: "
        print classifier
        print "\nTesting the results with the rest of the training data"
        # get a subsample to be faster
        local_to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*LOCAL_TEST_SUBSAMPLING)))
        local_to_keep = [i for i in local_to_keep if i not in to_keep]
        local_test_set_reduced = [training_set[i] for i in local_to_keep]
        # get prediction and output score
        local_test_features = feature_engineering(local_test_set_reduced, IDs, node_info, stemmer, stpwds, g)
        local_pred = classifier.predict(local_test_features)
        # get corresponding labels
        local_labels = [int(element[2]) for element in local_test_set_reduced]
        local_labels = list(local_labels)
        print f1_score(local_labels, local_pred)
        print "Testing done"

# ---Prediction--- #
if submission_mode:
    print "Creating features and prediction for the test set"
    # create test features
    testing_features = feature_engineering(testing_set, IDs, node_info, stemmer, stpwds, g)

    # issue predictions
    if probabilistic_mode:
        # average the probabilistic scores from all used classifiers
        avg_prob_predictions = np.array(np.zeros((len(testing_features),2)))
        for classifier in clfs:
            prob_predictions = np.array(list(classifier.predict_proba(testing_features)))
            prob_predictions = prob_predictions/len(clfs)
            avg_prob_predictions += prob_predictions

        # use median instead of 0.5 if we want 50/50 split between classes
        #predictions_med = [0 if x>np.median(avg_prob_predictions[:, 0]) else 1 for x in avg_prob_predictions[:, 0]]
        predictions_true = [0 if x>0.5 else 1 for x in avg_prob_predictions[:, 0]]

        #predictions_med = zip(range(len(testing_set)), predictions_med)
        predictions_true = zip(range(len(testing_set)), predictions_true)
        
    else:    
        predictions = list(classifier.predict(testing_features))
        predictions = zip(range(len(testing_set)), predictions)

    """
    with open(path_to_predictions + submission_name + "_predictions_prob.csv", "wb") as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(('ID','category'))
        for row in predictions_med:
            csv_out.writerow(row)
    """

    with open(path_to_predictions + submission_name + "_predictions.csv", "wb") as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(('ID','category'))
        for row in predictions_true:
            csv_out.writerow(row)
    print "Predictions done"
