import numpy as np
from sklearn import preprocessing
from gensim.models.word2vec import Word2Vec
import igraph
from graph_of_words import *


def clean(s, stemmer, stpwds):
    s = s.lower().split(" ")
    s = [token for token in s if token not in stpwds]
    s = [stemmer.stem(token) for token in s]
    s = [''.join([elt for elt in token if not elt.isdigit()]) for token in s] # remove digits
    s = [token for token in s if len(token)>2] # remove tokens shorter than 3 characters in size
    s = [token for token in s if len(token)<=25] # remove tokens exceeding 25 characters in size
    return s

def build_w2v(node_info, stemmer, stpwds):
    try:
        model = Word2Vec.load("w2v_model")
        print "Word2Vec model loaded"
    except:
        path_to_google_news = '/Users/jacob/Documents/School/INF582/'
        my_q = 300 # to match dim of GNews word vectors
        mcount = 5
        model = Word2Vec(size=my_q, min_count=mcount)
        cleaned_abstracts = [clean(element[5], stemmer, stpwds) for element in node_info]
        print "Building Word2Vec vocab..."
        model.build_vocab(cleaned_abstracts)
        print "Loading intersect vectors..."
        model.intersect_word2vec_format(path_to_google_news + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
        model.save("w2v_model")
        print "Model saved to disk"
    return model

def count_authLinksStoT (information_set, node_info):
    authLinks = {}
    for i in xrange(len(information_set)):
        source = information_set[i][0]
        target = information_set[i][1]

        source_info = [element for element in node_info if element[0] == source][0]
        target_info = [element for element in node_info if element[0] == target][0]

        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")
        
        for s in source_auth:
            s.replace(' ', '')
        for t in target_auth:
            t.replace(' ', '')
        
        for s in source_auth:
            for t in target_auth:
                key = (s,t)
                if key in authLinks:
                    authLinks[key] += 1
                else:
                    authLinks[key] = 1
    return authLinks


def feature_engineering(information_set, IDs, node_info, stemmer, stpwds, g, pairwise_similarity):
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
    # number of common neighbors
    num_common_neighbors = []

    # number of keywords: graph of words
    num_keywords_graph_of_words = []

    # TF_IDF
    pairwise_similarity_number = []

    w2v = build_w2v(node_info, stemmer, stpwds)

    # the average number of citations the authors of target have received from authors of source
    avg_number_citations_of_authors = []   
    # Authors link counter
    authLinks = count_authLinksStoT(information_set, node_info)

    counter = 0

    degrees = g.degree(IDs)
    neighbors_list = []
    for id in IDs:
        neighbors_list.append(set(g.neighbors(id)))

    print len(information_set), "examples to process:"
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
        wmd.append(w2v.wv.wmdistance(source_abstract, target_abstract))
        num_references_source.append(degrees[index_source])
        num_references_target.append(degrees[index_target])
        num_common_neighbors.append(len(neighbors_list[index_source].intersection(neighbors_list[index_target])))

        num_keywords_graph_of_words.append(len(set(keywords_graph_of_words(source_abstract)).intersection(set(keywords_graph_of_words(target_abstract)))))
       # print pairwise_similarity.shape
     #   pairwise_similarity_number.append(pairwise_similarity[index_source, index_target])

        # Count the average number of citations the authors of target have received from authors of source
        summ = 0
        count = 0
        for s in source_auth:
            for t in target_auth:
                key = (s,t)
                if key in authLinks:
                    summ += authLinks[key]
                    count += 1
        if count == 0:
            avg_number_citations_of_authors.append(0)
        else:
            avg_number_citations_of_authors.append(summ/count)

        counter += 1
        if counter % 1000 == 0:
            print counter, "examples processed"

    list_of_features = []
    list_of_features.append(overlap_title)
    list_of_features.append(temp_diff)
    list_of_features.append(comm_auth)
    list_of_features.append(wmd)
    list_of_features.append(num_references_source)
    list_of_features.append(num_references_target)
    list_of_features.append(num_common_neighbors)
    list_of_features.append(avg_number_citations_of_authors)
    list_of_features.append(num_keywords_graph_of_words)
    list_of_features.append(pairwise_similarity_number)
    # convert list of lists into array
    # documents as rows, unique words as columns (i.e., example as rows, features as columns)
    features = np.array(list_of_features).T
    # scale
    features = preprocessing.scale(features)
    return features
