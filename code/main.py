import random
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LogisticRegression
import nltk
import csv
import sys
import pickle
from sklearn.model_selection import KFold
from feature_engineering import *
from classifier_testing import *
from read_data import *
from graph_creation import *
from sklearn.ensemble import RandomForestClassifier


# ---Parameters--- #
submission_mode = False
testing_mode = False
quick_eval_mode = True
classifier_tuning_mode = False
probabilistic_mode = False
cv_on = True
submission_name = "0.05_g1and2_wmd_idf_auth_citation_cv"
TRAINING_SUBSAMPLING = 0.05
LOCAL_TEST_SUBSAMPLING = 0.05
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
path_to_data = "../data/"
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
pairwise_similarity = features_TFIDF * features_TFIDF.T
#print pairwise_similarity.shape
# ---Create graph--- #
g = create_graph(training_set, IDs)
#authors_citations_dictionary = []
# authors_citations_dictionary = create_authors_dictionary(training_set, node_info)
# ---Training--- #
print "Training"
# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*TRAINING_SUBSAMPLING)))
training_set_reduced = [training_set[i] for i in to_keep]
# create training features

# TODO delete
testing_features = feature_engineering(testing_set, IDs, node_info, stemmer, stpwds, g, pairwise_similarity)
np.save(path_to_data + 'testing_features100.npy', testing_features)
#testing_set = testing_set[:100] # TODO delete
if quick_eval_mode:
    print "Loading pre-trained features"
    training_features = np.load(path_to_data + 'training_features100.npy')
    testing_features = np.load(path_to_data + 'testing_features100.npy')
    labels_array = np.load(path_to_data + 'labels_array100.npy')
else:
    training_features = feature_engineering(training_set_reduced, IDs, node_info, stemmer, stpwds, g, pairwise_similarity)
    np.save(path_to_data + 'training_features_005.npy', training_features)
    testing_features = feature_engineering(testing_set, IDs, node_info, stemmer, stpwds, g, pairwise_similarity)
    np.save(path_to_data + 'training_features_005.npy', testing_features)
    # convert labels into integers then into column array
    labels = [int(element[2]) for element in training_set_reduced]
    labels = list(labels)
    labels_array = np.array(labels)
    np.save(path_to_data + 'labels_array_005.npy', labels_array)
    print "Features calculated"

# initialize classifier(s)
if classifier_tuning_mode:
    #tune_rf(train_x=training_features, train_y=labels_array,seed=seed)
    tune_SVC(train_x=training_features, train_y=labels_array)
    sys.exit(0)
else:
    clfs = []
    clfs.append(RandomForestClassifier(n_estimators=45, max_depth=25, min_samples_leaf=2, random_state=seed))
    clfs.append(svm.SVC(C=5, gamma=0.015, probability=True))
    clfs.append(LGBMClassifier())
    clfs_names = ["random_forest", "SVC", "LGBM"]


if cv_on:
    count_classifier = 0
    predictions_total = np.zeros(len(testing_set))
    for classifier in clfs:
        print "\n\nClassifier: "
        print classifier
        # Training
        # Create cross validation iterator
        kf = KFold(n_splits=5, random_state=1, shuffle=True)
        validation_scores = []
        predictions = np.zeros(len(testing_set))
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
            # Make test set prediction
            predictions += classifier.predict_proba(testing_features)[:, 1]
            cv_index += 1
        print "mean score: ", sum(validation_scores)/5
        predictions /= 5
        predictions_total += predictions
        predictions_true = [0 if x < 0.5 else 1 for x in predictions]
        # predictions_med = zip(range(len(testing_set)), predictions_med)
        predictions_true = zip(range(len(testing_set)), predictions_true)
        with open(path_to_predictions + submission_name + "_" + clfs_names[count_classifier] + "_predictions.csv", "wb") as pred1:
            csv_out = csv.writer(pred1)
            csv_out.writerow(('ID', 'category'))
            for row in predictions_true:
                csv_out.writerow(row)
        print "Predictions done"
        count_classifier += 1
    predictions_total /= 3
    predictions_true = [0 if x < 0.5 else 1 for x in predictions_total]
    # predictions_med = zip(range(len(testing_set)), predictions_med)
    predictions_true = zip(range(len(testing_set)), predictions_true)
    with open(path_to_predictions + submission_name + "_predictions_total.csv",
              "wb") as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(('ID', 'category'))
        for row in predictions_true:
            csv_out.writerow(row)
else:
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
            local_test_features = feature_engineering(local_test_set_reduced, IDs, node_info, stemmer, stpwds, g, pairwise_similarity)
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
        testing_features = feature_engineering(testing_set, IDs, node_info, stemmer, stpwds, g, pairwise_similarity)
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
