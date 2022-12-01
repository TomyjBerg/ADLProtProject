class ProteinGraph:

    '''Class storing the data of a ply file, including coordinates of the graph nodes (pos), 
    the features (x), and the adjacency information (edge_index)'''

    def __init__(self, features, pos, edge_index, name):
        self.features = features
        self.pos = pos
        self.edge_index = edge_index
        self.name = name

    def num_nodes(self):
        return len(self.pos)

    def num_features(self):
        return self.features.shape[1]

    def __str__(self):
        string = '\
            Number of Nodes: {n}\n\
            Features: {f}\n\
            Edge_Index: {a}\n\
            Coordinates of Nodes: {c}\n\
            Protein Name: {name}'\
            .format(n = len(self.features), f = self.features.shape, a = tuple(self.edge_index.shape), \
            c = self.pos.shape, name = self.name)
        return string