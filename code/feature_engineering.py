import numpy as np
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec
import igraph

def clean(s, stemmer, stpwds):
    s = s.lower().split(" ")
    s = [token for token in s if token not in stpwds]
    s = [stemmer.stem(token) for token in s]
    s = [''.join([elt for elt in token if not elt.isdigit()]) for token in s] # remove digits
    s = [token for token in s if len(token)>2] # remove tokens shorter than 3 characters in size
    s = [token for token in s if len(token)<=25] # remove tokens exceeding 25 characters in size
    return s

def build_w2v(node_info, stemmer, stpwds):
    path_to_google_news = '/Users/jacob/Documents/School/INF582/'
    my_q = 300 # to match dim of GNews word vectors
    mcount = 5
    model = Word2Vec(size=my_q, min_count=mcount)
    cleaned_abstracts = [clean(element[5], stemmer, stpwds) for element in node_info]
    model.build_vocab(cleaned_abstracts)
    model.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
    return model

def feature_engineering(information_set, IDs, node_info, stemmer, stpwds, g):
    # number of overlapping words in title
    overlap_title = []
    # temporal distance between the papers
    temp_diff = []
    # number of common authors
    comm_auth = []
    # WMD
    wmd = []
    # number of references for the source or the target
    num_references_source = []
    num_references_target = []

    # w2v = build_w2v(node_info, stemmer, stpwds)

    counter = 0

    degrees = g.degree(IDs)

    for i in xrange(len(information_set)):
        source = information_set[i][0]
        target = information_set[i][1]

        index_source = IDs.index(source)
        index_target = IDs.index(target)

        source_info = [element for element in node_info if element[0] == source][0]
        target_info = [element for element in node_info if element[0] == target][0]

        source_title = clean(source_info[2], stemmer, stpwds)
        target_title = clean(target_info[2], stemmer, stpwds)

        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")

        source_abstract = clean(source_info[5], stemmer, stpwds)
        target_abstract = clean(target_info[5], stemmer, stpwds)

        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
    #    wmd.append(w2v.wv.wmdistance(source_abstract, target_abstract))
        num_references_source.append(degrees[index_source])
        num_references_target.append(degrees[index_target])

        counter += 1
        if counter % 1000 == 0:
            print counter, "examples processed"

    list_of_features = []
    list_of_features.append(overlap_title)
    list_of_features.append(temp_diff)
    list_of_features.append(comm_auth)
    # list_of_features.append(wmd)
    list_of_features.append(num_references_source)
    list_of_features.append(num_references_target)
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    features = np.array(list_of_features).T
    # scale
    features = preprocessing.scale(features)
    return features
