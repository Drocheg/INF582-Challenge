import igraph
import pickle


def save_obj(obj, name ):
    with open('../'+ name + '.pkl', 'wb') as f:
        print "open!"
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_graph(edges_set, node_ids):
    """
    Creates a graph out of the information given
        :param edges_set: set of pair (origin id, target id) used to create edges
        :param node_ids: set of ids used to create nodes
        :return: the graph created
    """
    # look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas
    print "creating graph"
    edges = [(element[0],element[1]) for element in edges_set if element[2]=="1"]

    # some nodes may not be connected to any other node
    # hence the need to create the nodes of the graph from node_info.csv,
    # not just from the edge list
    nodes = node_ids

    # create empty directed graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(nodes)

    # add edges
    g.add_edges(edges)
    print "graph created"
    return g


def create_authors_dictionary(information_set, node_info):
    try:
        return load_obj("authors_citations_dictionary_reduced")
    except:
        print "error loading"
    finally:
        print "creating authors dictionary"
        authors_citations_dictionary = {}
        counter = 0
        print len(information_set), "information to process:"
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
                    key = (s, t)
                    if key in authors_citations_dictionary:
                        authors_citations_dictionary[key] += 1
                    else:
                        authors_citations_dictionary[key] = 1
            counter += 1
            if (counter + 1) % 1000 == 0:
                print counter, "info processed"
        print "authors dictionary created"
        try:
            save_obj(authors_citations_dictionary, "authors_citations_dictionary_reduced")
        finally:
            return authors_citations_dictionary

