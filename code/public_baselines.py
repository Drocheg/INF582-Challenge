import random
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import linear_kernel
import nltk
import csv
from feature_engineering import *
from read_data import *
from graph_creation import *

# ---First Initializations--- #
random.seed(0)  # to be able to reproduce results
path_to_predictions = "../predictions"
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
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.005)))
training_set_reduced = [training_set[i] for i in to_keep]
# create training features
training_features = feature_engineering(training_set_reduced, IDs, node_info, stemmer, stpwds)
# convert labels into integers then into column array
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)
# initialize basic SVM
classifier = svm.LinearSVC()
# train model with features and labels
classifier.fit(training_features, labels_array)
print "Training done"

# ---Test--- #
print "Testing the results with the rest of the training data"
# get a subsample to be faster
local_to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.005)))
local_to_keep = [i for i in local_to_keep if i not in to_keep]
local_test_set_reduced = [training_set[i] for i in local_to_keep]
# get prediction and output score
local_test_features = feature_engineering(local_test_set_reduced, IDs, node_info, stemmer, stpwds)
local_pred = classifier.predict(local_test_features)
# get corresponding labels
local_labels = [int(element[2]) for element in local_test_set_reduced]
local_labels = list(local_labels)
print f1_score(local_labels, local_pred)
print "Testing done"

# ---Prediction--- #
print "Creating features and prediction for the test set"
# create test features
testing_features = feature_engineering(testing_set, IDs, node_info, stemmer, stpwds)
# issue predictions
predictions_SVM = list(classifier.predict(testing_features))
# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)
with open(path_to_predictions + "improved_predictions.csv", "wb") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)
print "Predictions done"
