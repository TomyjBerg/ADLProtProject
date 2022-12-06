from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.typing import PairTensor
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
import random 
import numpy as np
import pandas as pd
from c_ProteinGraph import ProteinGraph
from c_GraphPatch import GraphPatch
from helper_functions import save_object



def k_subgraph_perso(
    node_idx: Union[int, List[int], Tensor],
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    max_nodes = int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    unique_new_subsets = [node_idx]

    for i in range(9999):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        unique_new_subsets.append(subset)
        if len(unique_new_subsets[i+1]) > max_nodes:
            removing = len(unique_new_subsets[i+1]) - max_nodes
            not_in_list = []
            for e in unique_new_subsets[i+1]:
                if e not in unique_new_subsets[i]:
                    e = int(e)
                    not_in_list.append(e)      
            remove_element = random.sample(not_in_list, k=removing)
            list_np = torch.Tensor.numpy(unique_new_subsets[i+1])
            list_list = list(list_np)
            for k in remove_element:
                list_list.remove(k)
            list_np = np.asarray(list_list)
            subset = torch.from_numpy(list_np)
            break
        
    
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes, ), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index_relab = node_idx[edge_index]
    


    return subset, edge_index, edge_index_relab,inv, edge_mask


def create_patches(subset,edge_index_relab,prot,lab):
    features = prot.features.iloc[list(np.asarray(subset))]
    features_patches = features[['charge','hbond','hphob']]
    features_patches = torch.Tensor(np.asarray(features_patches))
    sub_idx = subset
    sub_edge_index = edge_index_relab
    names = prot.name
    patch = GraphPatch(features_patches,sub_idx,sub_edge_index,lab,names)
    return patch

def match_iface(sub,iface,match_criteria):
    compteur = 0
    sub = list(np.asarray(sub))
    for i in range(len(sub)):
        if sub[i] not in iface:
            compteur = compteur+1
    perc = compteur/len(sub)
    if perc > match_criteria:
        return False
    else:
        return True


def extract_patches(prot_complex,max_graph_size,number_wanted,match_criteria,file_lab_0,file_lab_1):
    
    sub_iface_complex = []
    edge_index_iface_relab_complex = []
    for prot in prot_complex:
        center_iface = prot.iface_center[0]
        sub_iface,edge_index_iface,edge_index_iface_relab,_,_  = k_subgraph_perso(center_iface,
                                                            prot.edge_index,max_nodes=max_graph_size)
        sub_iface_complex.append(sub_iface)
        edge_index_iface_relab_complex.append(edge_index_iface_relab)

    if match_iface (sub_iface_complex[0],prot_complex[0].iface_idx,match_criteria) & match_iface(sub_iface_complex[1],prot_complex[1].iface_idx,match_criteria):
        for i in range(len(prot_complex)):
            patch = create_patches(sub_iface_complex[i],edge_index_iface_relab_complex[i],prot_complex[i],1)
            file_lab_1_A = file_lab_1 + '/' + prot_complex[i].name
            save_object(patch,file_lab_1_A)
            print(prot_complex[i].name + ' lab ' + str(1))
    for prot in prot_complex:
        file_lab_0_A = file_lab_0 + '/' + prot.name
        for i in range(number_wanted):
            match_bool = True
            while match_bool == True:
                center_node = random.randint(0,prot.features.shape[0]-1)
                subset,edge_index,edge_index_relab,_,_= k_subgraph_perso(center_node,prot.edge_index,max_nodes=max_graph_size)
                match_bool = match_iface (subset,prot.iface_idx,match_criteria)
            patch = create_patches(subset,edge_index_relab,prot,0)
            filename_0 = file_lab_0_A + '_' + str(i)
            save_object(patch,filename_0)
            print(prot.name + ' lab ' + str(2))    

