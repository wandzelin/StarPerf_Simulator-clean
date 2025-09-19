import numpy as np
import networkx as nx
import h5py
from math import radians, cos, sin, asin, sqrt
from select_satellite import select_nearest



def delay(constellation_name , source , target , shell_name , t):
    file_path = "data/XML_constellation/" + constellation_name + ".h5"  # h5 file path and name

    with h5py.File(file_path, 'r') as file:
        
        delay_group = file['delay']
        current_shell_group = delay_group[shell_name]
        delay = np.array(current_shell_group['timeslot' + str(t)]).tolist()


    G = nx.Graph() 
    satellite_nodes = []
    for i in range(1 , len(delay) , 1):
        satellite_nodes.append("satellite_" + str(i))
    G.add_nodes_from(satellite_nodes)  

    satellite_edges = []
    edge_weights = []
    for i in range(1 , len(delay) , 1):
        for j in range(i+1 , len(delay) , 1):
            if delay[i][j] > 0:
                satellite_edges.append(("satellite_" + str(i) , "satellite_" + str(j) , delay[i][j]))
                edge_weights.append(delay[i][j])
    
    G.add_weighted_edges_from(satellite_edges)

    start_satellite = "satellite_" + str(source.id)  # 起始路由卫星的编号
    end_satellite = "satellite_" + str(target.id)  # 终止路由卫星的编号

    t_minimum_delay_time = nx.dijkstra_path_length(G, source=start_satellite, target=end_satellite)
    
    return t_minimum_delay_time * 2


