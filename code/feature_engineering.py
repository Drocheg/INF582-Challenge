import numpy as np
from sklearn import preprocessing

def feature_engineering(information_set, IDs, node_info, stemmer, stpwds):
    # number of overlapping words in title
    overlap_title = []
    # temporal distance between the papers
    temp_diff = []
    # number of common authors
    comm_auth = []

    counter = 0
    for i in xrange(len(information_set)):
        source = information_set[i][0]
        target = information_set[i][1]

        index_source = IDs.index(source)
        index_target = IDs.index(target)

        source_info = [element for element in node_info if element[0] == source][0]
        target_info = [element for element in node_info if element[0] == target][0]

        source_title = source_info[2].lower().split(" ")
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]

        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]

        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")

        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

        counter += 1
        if counter + 1 % 1000 == True:
            print counter, "examples processed"

    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    features = np.array([overlap_title, temp_diff, comm_auth]).T
    # scale
    features = preprocessing.scale(features)
    return features
