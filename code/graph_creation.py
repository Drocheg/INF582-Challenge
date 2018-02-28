def create_graph():
    return 0
    ## the following shows how to construct a graph with igraph
    ## even though in this baseline we don't use it
    ## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

    # edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

    ## some nodes may not be connected to any other node
    ## hence the need to create the nodes of the graph from node_info.csv,
    ## not just from the edge list

    # nodes = IDs

    ## create empty directed graph
    # g = igraph.Graph(directed=True)

    ## add vertices
    # g.add_vertices(nodes)

    ## add edges
    # g.add_edges(edges)

