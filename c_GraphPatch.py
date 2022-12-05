import numpy as np

class GraphPatch:

    '''Class storing the data of an extracted surface patch, including the features of the nodes,
    the indexes of the nodes (from the whole protein), the edge_index, the label of the patch and 
    the name of the protein it has been taken from: Save data in this class as follows: 
    
    from c_GraphPatch import GraphPatch
    name_of_object = GraphPatch(features, indexes, edge_index, label, name)

    It would be best if all inputs were already torch tensors with dtype float32 (Except the name)
    '''

    def __init__(self, features, indexes, edge_index, label, name):
        self.x = features
        self.indexes = indexes
        self.edge_index = edge_index
        self.y = label
        self.name = name

    def num_nodes(self):
        return self.features.shape[0]

    def num_edges(self):
        return self.edge_index.shape[1]

    def num_features(self):
        return self.features.shape[1]

    def __str__(self):
        string = '\
            Number of Nodes: {n}\n\
            Features: {f}\n\
            Edge Index: {i}\n\
            Fitness: {y}\n\
            Protein Name: {name}'\
            .format(n = self.features.shape[0], f = self.features.shape, i = self.edge_index.shape, y=self.y, name = self.name)
        return string