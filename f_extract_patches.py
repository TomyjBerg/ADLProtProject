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


    return subset, edge_index, inv, edge_mask


def create_patches(subset,edge_index,features):
    features_patches = features.iloc[list(np.asarray(subset))]
    patch = [subset,edge_index,features_patches]
    return patch

def extract_patches(prot,max_graph_size,number_wanted):
    patches = ()
    for i in range(number_wanted):
        center_node = random.randint(0,prot.features.shape[0])
        subset,edge_index,mapping,edge_mask = k_subgraph_perso(center_node,prot.edge_index,max_nodes=max_graph_size)
        patch = create_patches(subset,edge_index,prot.features)
        patches.append(patch)
    return patches
    
    

