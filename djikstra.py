# Dijkstra algorithm implementation for a graph that is saved as a dictionary

# {graph_node_index:{neighbor1:distance, neighbor2:distance]

def distances_from_center(graph, center):
    
    '''Function that takes a graph and the starting node and returns a list of distances 
    from the starting node to every other node'''
    
    # INITIALIZE THE DICTIONARY TO SAVE ALL DISTANCES TO THE CENTER NODE
    dist_from_center = {key:100 for key in graph}
    dist_from_center[center] = 0
    unseen_nodes = list(dist_from_center.keys())
    
    for _ in graph:

        # IDENTIFICATION OF THE NEXT POINT TO LOOK AT (SHORTES DISTANCE FROM START)
        dist = 101
        for node in unseen_nodes:
            if dist_from_center[node]<dist:
                dist = dist_from_center[node]
                loc = node

        # LOOP THROUGH ALL THE NEIGHBORS OF THE LOC NODE AND ADJUST THE VALUES OF THOSE, IF NEEDED
        for neighbor, weight in graph[loc].items():               
            if dist + weight < dist_from_center[neighbor]:
                dist_from_center[neighbor] = dist + weight 
        unseen_nodes.remove(loc)
        
    return dist_from_center