import igraph


def create_graph(edges_set, node_ids):
    """
    Creates a graph out of the information given
        :param edges_set: set of pair (origin id, target id) used to create edges
        :param node_ids: set of ids used to create nodes
        :return: the graph created
    """
    # look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

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

    return g
