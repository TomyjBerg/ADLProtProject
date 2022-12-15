import numpy as np
import random
import os
import torch
import fnmatch
from torch_geometric.data import Dataset
from helper_functions import load_object
from torch_geometric.data import Data
#from torch_geometric.utils import to_dense_adj
import torch.nn.functional as f


class PatchDataset(Dataset):

    '''Custom dataset for generation of datasets of graphs extracted from protein surfaces.
    The input data directories should contain instances of the GraphPatch class stored as pkl files with the characters
    0-4 of the filename indicating the complex name. The function get_item returns an instance of the
    torch_geometric.Data class which contains a couple of two graphs with the following information:  
    - node features x
    - adjacency matrix
    - label of the graph
    '''

    def __init__(self, data_dir_label_0, data_dir_label_1, neg_pos_ratio):
        
        self.dir_lab0 = data_dir_label_0
        self.dir_lab1 = data_dir_label_1

        lab0_list = os.listdir(self.dir_lab0)
        lab1_list = os.listdir(self.dir_lab1)

        # Create a list of the complex names
        names = [i[0:4] for i in lab1_list]
        self.complexes = [i[0:4] for j, i in enumerate(names) if i[0:4] not in names[:j]]

        couples = []

        # create random couples from the lab0 directory
        for _ in range(len(self.complexes)*neg_pos_ratio):
            prot1, prot2 = random.sample(lab0_list, 2)
            couples.append((0, prot1, prot2))
        
        # create couples from the lab1 directory according to the file names
        for comp in self.complexes: 
            filenames = fnmatch.filter(lab1_list, comp+'*')
            couples.append((1, filenames[0], filenames[1]))

        random.shuffle(couples)
        self.couples = couples


    def __len__(self):
        return len(self.couples)

    def __getitem__(self, idx):

        # generate the paths to the two patches
        positive, file1, file2 = self.couples[idx]

        if positive == 1:
            path1 = self.dir_lab1 + '/' + file1
            path2 = self.dir_lab1 + '/' + file2
            patch1 = load_object(path1)
            patch2 = load_object(path2)
        
        if positive == 0: 
            path1 = self.dir_lab0 + '/' + file1
            path2 = self.dir_lab0 + '/' + file2
            patch1 = load_object(path1)
            patch2 = load_object(path2)


        x1 = patch1.x.float()
        x1 = f.normalize(x1,dim=1)
        x1[:,1] = x1[:,1]*(-1)
        x1[:,0] = x1[:,0]*(-1)
        y1 = torch.tensor(patch1.y).long()
        edge_index1 = patch1.edge_index.long()
        #adj1 = torch.squeeze(to_dense_adj(edge_index1))

        x2 = patch2.x.float()
        x2 = f.normalize(x2,dim=1)
        y2 = torch.tensor(patch2.y).long()
        edge_index2 = patch2.edge_index.long()
        #adj2 = torch.squeeze(to_dense_adj(edge_index2))   

        return Data(edge_index = edge_index1, x=x1, y=torch.tensor(positive).long()), Data(edge_index = edge_index2, x=x2, y=torch.tensor(positive).long())
