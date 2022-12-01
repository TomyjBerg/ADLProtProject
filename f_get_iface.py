def get_iface(pos, features):

    '''Function that takes as input the coordinates of all surface nodes of a protein (numpy array)
    and the features of the nodes (pandas df), including an iface value. Returns a list of indeces indicating 
    which nodes are part of the interface and also returns the center of the interface region'''

    # Split pos and features into two df (one iface, one not iface)
    iface_idx = features.index[features['iface'] == 1.0].tolist()
    not_iface_idx = features.index[features['iface'] == 0].tolist()

    iface_pos = pos[iface_idx]
    not_iface_pos = pos[not_iface_idx]

    from sklearn.neighbors import NearestNeighbors
    
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(not_iface_pos) 

    largest_dist = 0
    for i in range(len(iface_pos)): 
        if (dist := neigh.kneighbors([iface_pos[i]])[0][0][0]) > largest_dist:
            largest_dist = dist
            center = i, iface_idx[i], iface_pos[i]

    return iface_idx, center 