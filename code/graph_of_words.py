import re
import itertools
import operator
import copy
import igraph
import heapq
import nltk
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag


def terms_to_graph(terms, w):
    '''This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox'].
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'.
    '''

    from_to = {}

    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = []

    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
# error
    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        considered_term = terms[i]  # term to consider
        terms_temp = terms[(i - w + 1):(i + 1)]  # all terms within sliding window

        # edges to try
        candidate_edges = []
        for p in range(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            if try_edge[1] != try_edge[0]:
                # if not self-edge

                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(sorted(set(terms)))

    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())

    # set edge and vertex weights
    g.es['weight'] = from_to.values()  # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=from_to.values())  # weighted degree

    return (g)


def unweighted_k_core(g):
    # work on clone of g to preserve g
    gg = copy.deepcopy(g)

    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs['name'], [0] * len(gg.vs)))

    i = 0

    # while there are vertices remaining in the graph
    while len(gg.vs) > 0:
        # while there is a vertex with degree less than i
        while [deg for deg in gg.strength() if deg <= i]:
            index = [ind for ind, deg in enumerate(gg.strength()) if deg <= i][0]
            # assign i as the matching vertices' core numbers
            cores_g[gg.vs[index]['name']] = i
            gg.delete_vertices(index)  # incident edges on the deleted vertex are automatically removed

        i += 1

    return cores_g


def keywords_graph_of_words(abstracts_cleaned):
    # create graph-of-words with window of size 4
    g = terms_to_graph(abstracts_cleaned, w=4)
    # decompose graph-of-words
    core_numbers = unweighted_k_core(g)
    # retain main core as keywords
    max_c_n = max(core_numbers.values())
    return [kwd for kwd,c_n in core_numbers.iteritems() if c_n==max_c_n]
