import random
import numpy as np
import igraph
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
import csv
from feature_engineering import *

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()

path_to_data = "../data/"

with open(path_to_data + "testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

###################
# random baseline #
###################

random_predictions = np.random.choice([0, 1], size=len(testing_set))
random_predictions = zip(range(len(testing_set)), random_predictions)

with open(path_to_data + "random_predictions.csv", "wb") as pred:
    csv_out = csv.writer(pred)
    for row in random_predictions:
        csv_out.writerow(row)
        
# note: Kaggle requires that you add "ID" and "category" column headers

###############################
# beating the random baseline #
###############################

# the following script gets an F1 score of approximately 0.66

# data loading and preprocessing 

# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes

# ---Read Data--- #
with open(path_to_data + "training_set.txt", "r") as f:
    reader = csv.reader(f)
    training_set = list(reader)
training_set = [element[0].split(" ") for element in training_set]
with open(path_to_data + "node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)
IDs = [element[0] for element in node_info]

# ---compute TFIDF vector of each paper--- #
corpus = [element[5] for element in node_info]
vectorizer = TfidfVectorizer(stop_words="english")
# each row is a node in the order of node_info
features_TFIDF = vectorizer.fit_transform(corpus)

# ---create graph--- #
#g = create_graph()

# ---training--- #
# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set)*0.025)))
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

# ---test--- #
# create test features
testing_features = feature_engineering(testing_set, IDs, node_info, stemmer, stpwds)
# issue predictions
predictions_SVM = list(classifier.predict(testing_features))
# write predictions to .csv file suitable for Kaggle (just make sure to add the column names)
predictions_SVM = zip(range(len(testing_set)), predictions_SVM)
with open(path_to_data + "improved_predictions.csv", "wb") as pred1:
    csv_out = csv.writer(pred1)
    for row in predictions_SVM:
        csv_out.writerow(row)

