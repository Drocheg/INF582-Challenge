import random
import numpy as np
import nltk
import csv
import sys
import pickle
from feature_engineering import *
from classifier_testing import *
from read_data import *
from graph_creation import *
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# ---Parameters--- #
submission_mode = False         # if a submission file should be created
testing_mode = False            # if we want to evaluate local estimation of score
quick_eval_mode = True          # if we want to load features from files rather than compute
classifier_tuning_mode = False  # if we're tuning classifier hyperparameters to find optimal settings
cv_on = True                    # whether to use cross-validation for the local score
probabilistic_mode = False      # if we want to average probabilistic score from multiple classifiers
                                #   note: automatically done if cv_on

submission_name = "0.05_g1and2_wmd_idf_auth_citation_cv"
TRAINING_SUBSAMPLING = 0.05     # subset to train on, if computing features
LOCAL_TEST_SUBSAMPLING = 0.025  # subset for local test score
seed = 1337

# ---First Initializations--- #
random.seed(seed)               # to be able to reproduce results
path_to_predictions = "../predictions/"
path_to_data = "../data/"
nltk.download('punkt')          # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

# ---Read Data--- #
testing_set, training_set, node_info = read_data()

IDs = [element[0] for element in node_info]

# ---Compute TFIDF vector of each paper--- #
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info

pairwise_similarity = [] #features_TFIDF * features_TFIDF.T

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

if quick_eval_mode:
    print "Loading pre-trained features"
    training_features = np.load(path_to_data + 'training_features100.npy')
    testing_features = np.load(path_to_data + 'testing_features100.npy')
    labels_array = np.load(path_to_data + 'labels_array100.npy')
    training_auth_feature = np.array([np.load(path_to_data + 'avg_auth_train.npy').squeeze()]).T # separate loading because it was
    testing_auth_feature = np.array([np.load(path_to_data + 'avg_auth_test.npy').squeeze()]).T   # created later than the others,
    training_features = np.concatenate((training_features, training_auth_feature), axis=1)       # to avoid recomputing all
    testing_features = np.concatenate((testing_features, testing_auth_feature), axis=1)

    scaler = StandardScaler()       # scaling all input features, neccessary for some models
    scaler.fit(training_features)
    training_features = scaler.transform(training_features)
    testing_features = scaler.transform(testing_features)

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

# ---Classifiers--- #
# tune classifiers and then shut down, or initialize classifier(s) if not in tuning mode
if classifier_tuning_mode:
    #tune_rf(train_x=training_features, train_y=labels_array,seed=seed)
    #tune_SVC(train_x=training_features, train_y=labels_array)
    tune_lgbm(train_x=training_features, train_y=labels_array)
    sys.exit(0)
else:
    clfs = []
    clfs.append(MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(12, 12), random_state=seed, verbose=10))
    #clfs.append(RandomForestClassifier(n_estimators=45, max_depth=25, min_samples_leaf=2, random_state=seed))
    clfs.append(LGBMClassifier(num_leaves=127, reg_alpha=0.5, max_depth=8, min_data_in_leaf=16))
    #clfs.append(svm.SVC(C=16, gamma=0.125, probability=True))
    #clfs_names = ["random_forest", "SVC", "LGBM"]
    clfs_names = ["MLP", "LGBM"]

# ---Evaluation--- #
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
        print "mean score: ", sum(validation_scores)/5 # average the score over the folds
        predictions /= 5
        predictions_total += predictions
        predictions_true = [0 if x < 0.5 else 1 for x in predictions]
        predictions_true = zip(range(len(testing_set)), predictions_true)

        # ---Write to submission file for single classifier--- #
        with open(path_to_predictions + submission_name + "_" + clfs_names[count_classifier] + "_predictions.csv", "wb") as pred1:
            csv_out = csv.writer(pred1)
            csv_out.writerow(('ID', 'category'))
            for row in predictions_true:
                csv_out.writerow(row)
        print "Predictions done"
        count_classifier += 1

    predictions_total /= len(clfs_names) # average over all classifiers
    predictions_true = [0 if x < 0.5 else 1 for x in predictions_total]
    predictions_true = zip(range(len(testing_set)), predictions_true)

    # ---Write to submission file for ensemble--- #
    with open(path_to_predictions + submission_name + "_predictions_total.csv",
              "wb") as pred1:
        csv_out = csv.writer(pred1)
        csv_out.writerow(('ID', 'category'))
        for row in predictions_true:
            csv_out.writerow(row)

else:
    # train model with features and labels
    for classifier in clfs:
        print "Training single classifier..."
        classifier.fit(training_features, labels_array)
        print "Done"
    print "Training done"

    # ---Test--- #
    if testing_mode:

        # get a subsample to be faster
        local_to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*LOCAL_TEST_SUBSAMPLING)))
        local_to_keep = [i for i in local_to_keep if i not in to_keep]
        local_test_set_reduced = [training_set[i] for i in local_to_keep]
        # get prediction and output score
        print "Calculating local test features"
        local_test_features = feature_engineering(local_test_set_reduced, IDs, node_info, stemmer, stpwds, g, pairwise_similarity)
        # get corresponding labels
        local_labels = [int(element[2]) for element in local_test_set_reduced]
        local_labels = list(local_labels)

        # average the probabilistic scores from all used classifiers
        avg_prob_predictions = np.array(np.zeros((len(local_test_features),2)))

        for classifier in clfs:
            print "\n\nClassifier: "
            print classifier
            print "\nTesting the results with the rest of the training data"

            prob_predictions = np.array(list(classifier.predict_proba(local_test_features)))
            prob_predictions = prob_predictions/len(clfs)
            avg_prob_predictions += prob_predictions

            local_pred = classifier.predict(local_test_features)

            print f1_score(local_labels, local_pred)

        ensemble_predictions = [0 if x>0.5 else 1 for x in avg_prob_predictions[:, 0]]
        print "\n\nEnsemble of all classifiers: "
        print f1_score(local_labels, ensemble_predictions)


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

            predictions_true = [0 if x>0.5 else 1 for x in avg_prob_predictions[:, 0]]
            predictions_true = zip(range(len(testing_set)), predictions_true)

        else:
            predictions = list(classifier.predict(testing_features))
            predictions = zip(range(len(testing_set)), predictions)

        with open(path_to_predictions + submission_name + "_predictions.csv", "wb") as pred1:
            csv_out = csv.writer(pred1)
            csv_out.writerow(('ID','category'))
            for row in predictions_true:
                csv_out.writerow(row)
        print "Predictions done"
