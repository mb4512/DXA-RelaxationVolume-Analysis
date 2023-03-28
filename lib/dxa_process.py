import numpy as np
from copy import deepcopy
from scipy import spatial

class Node:
    '''Container for dislocation node data.'''
    def __init__(self):
        self.edges = np.array([], dtype=np.int)
        self.signs = np.array([], dtype=np.int)
        self.bvecs = np.array([[]], dtype=np.float)
        self.sgxyz = np.array([], dtype=np.int)
        
    def net(self):
        return np.dot(self.bvecs.T, self.signs)
    
    def sense(self):
        if -1 in self.signs and 1 in self.signs:
            return True
        return False

def nodes_register(rfile, de11, de12, de21, de22):
    '''Built a register of all nodes and their connectivity using segment connectivity matrix.'''
    nodes_start = []
    nseg = rfile.nsegments

    # nodes are defined by segment start and end points, though they may appear multiple times.
    # first, all nodes coming from start and then end points are collected independently.
    for i in range(nseg):
        node = Node()

        edges = np.where(de11[i])[0]
        node.edges = np.r_[[i], edges]
        node.signs = np.full_like(node.edges, -1)
        node.bvecs = np.r_[[rfile.btrue[i]], rfile.btrue[edges]]

        edges = np.where(de12[i])[0]
        node.edges = np.r_[node.edges, edges]
        node.signs = np.r_[node.signs, np.full_like(edges, 1)]
        node.bvecs = np.r_[node.bvecs, rfile.btrue[edges]]
        
        node.sgxyz = np.full_like(node.edges, 1)
        nodes_start.append(node)
        
    nodes_end = []
    for i in range(nseg):
        node = Node()

        edges = np.where(de22[i])[0]
        node.edges = np.r_[[i], edges]
        node.signs = np.full_like(node.edges, 1)
        node.bvecs = np.r_[[rfile.btrue[i]], rfile.btrue[edges]]
        
        edges = np.where(de21[i])[0]
        node.edges = np.r_[node.edges, edges]
        node.signs = np.r_[node.signs, np.full_like(edges, -1)]
        node.bvecs = np.r_[node.bvecs, rfile.btrue[edges]]

        node.sgxyz = np.full_like(node.edges, 1)
        nodes_end.append(node)

    # next, duplicated entries are removed from the total node register
    nodes_register = nodes_start + nodes_end

    # maximum edge segment id
    maxedge = np.max([np.max(node.edges) for node in nodes_register])

    # loop over all edges in the system
    ncount=0
    for i in range(maxedge):
        
        # find all nodes to which edge i is connected to
        sm=np.argwhere([(i==node.edges).any() for node in nodes_register])
       
        # raise warning if an edge is connected to more than two nodes (a consequence of duplicate nodes) 
        if sm.size>2:
            if ncount < 3:
                print ("Edge %d is connected to more than two nodes: " % i, sm)
            elif ncount == 4:
                print ("... surpressing further messages.")
            ncount +=1
            
    print ("Before: %d edges are connected to more than 3 nodes" % ncount) 
            
    # filter out duplicate nodes
    print("\nRemoving duplicate nodes.\n") 
    pm = lambda x: 'p' if x==1 else 'm'
    indices_sorted = []
    for node in nodes_register:

        strkey = np.sort(['%s%s' % (pm(j),i) for i,j in zip(node.edges,node.signs)])
        strkey = 'x'.join(strkey)
        indices_sorted.append(strkey)
        
    _,indices_unique = np.unique(indices_sorted, return_index=True)
    nodes_register = [nodes_register[_v] for _v in indices_unique]

    ncount = 0
    for i in range(maxedge):
        sm=np.argwhere([(i==node.edges).any() for node in nodes_register])
        if sm.size>2:
            print ("Edge %d is connected to more than two nodes: " % i, sm)
            ncount += 1

    print ("After: %d edges are connected to more than 3 nodes" % ncount)
    if ncount == 0:
        print ("No edges are connected to more than 2 nodes, internal check passed.")
        return nodes_register
    else:
        return 0

    #import copy
    #nodes_copy = copy.deepcopy(nodes_register)

def make_dictionaries(nodes_reg):
    '''Dictionary of edges and nodes for quick lookup.'''
    segment_ids = np.unique(np.concatenate([node.edges for node in nodes_reg]))

    edge_dict = {}
    _checked = []
    for ni,node in enumerate(nodes_reg):
        for ei,edge in enumerate(node.edges):
            
            if edge not in _checked:
                _checked += [edge]
                sign = node.signs[ei]
                bvec = node.bvecs[ei]

                for ni2,node2 in enumerate(nodes_reg):
                    for ei2,edge2 in enumerate(node2.edges):

                        if edge == edge2:
                            if sign == -1: # not sure why this way around; convention?
                                edge_dict[edge] = [[ni, ni2], bvec]
                            else:
                                edge_dict[edge] = [[ni2, ni], bvec]                            
                        continue
                    continue


    node_dict = {}
    for _skey in edge_dict.keys():
        node_dict[_skey] = []
        node_dict[_skey] = []

    for _skey in edge_dict.keys():
        node_dict[_skey] += [edge_dict[_skey][0][0]]
        node_dict[_skey] += [edge_dict[_skey][0][1]]
        
    return edge_dict, node_dict


def link_network(rfile, dmax=15.0, djoin=1.0e-3):
    '''Script for connecting terminating dislocation segments to neighbours.
    
    Loop over the ends of all dislocation segments and build a connectivity
    matrix. If dislocation segments terminate without connections, search
    for the nearest neighbouring dislocation segment to connect them to. 
    
    Arguments
    ---------
    rfile: class instance from ReadFile 
    
    Parameters
    ----------
    djoin: segment ends closer than this are considered as connected (default: 1.0e-3)
    dmax: maximum neighbour search radius in Angstrom (default: 15.0)
    
    Returns
    -------
    result: list of connectivity matrices between segment starts and ends 
    ''' 

    print ("Connect all loose dislocation segments and build a connectivity matrix.")
    
    # wrap ends of dislocation segments back into the box
    ends1 = np.r_[[x[0] for x in rfile.xyz]]
    endsw1 = np.r_[[_r - rfile.cell*np.floor(_r/rfile.cell) for _r in ends1]]
    
    ends2 = np.r_[[x[-1] for x in rfile.xyz]]
    endsw2 = np.r_[[_r - rfile.cell*np.floor(_r/rfile.cell) for _r in ends2]]
    
    # for segments that terminate, find neighboring segments to connect them to
    nseg = rfile.nsegments
    
    de11 = np.zeros((nseg, nseg))
    de22 = np.zeros((nseg, nseg))
    
    de12 = np.zeros((nseg, nseg))
    de21 = np.zeros((nseg, nseg))
    
    ktree1 = spatial.cKDTree(endsw1, boxsize=rfile.cell)
    ktree2 = spatial.cKDTree(endsw2, boxsize=rfile.cell)
    
    terminating = 0
    for i in range(len(endsw1)):    
        nnebs = 0

        # first, look for neighboring nodes right on top of the current node
        dis1,nebs1 = ktree1.query(endsw1[i], distance_upper_bound=djoin, k=8)
        dis2,nebs2 = ktree2.query(endsw1[i], distance_upper_bound=djoin, k=8)

        dis1  =  dis1[nebs1 != i]  # remove self interaction
        nebs1 = nebs1[nebs1 != i]

        nebs1 = nebs1[dis1<np.inf]
        nebs2 = nebs2[dis2<np.inf]

        # if no node is found, the segment terminates. look for the closest segment end
        if nebs1.size + nebs2.size == 0:
            
            dis1,nebs1 = ktree1.query(endsw1[i], distance_upper_bound=dmax, k=2)
            dis2,nebs2 = ktree2.query(endsw1[i], distance_upper_bound=dmax, k=1)
            
            dis1  = dis1[1] # remove self interaction
            nebs1 = nebs1[1]

            if dis1 < dis2:
                nebs2 = np.r_[[]]
                nebs1 = np.r_[nebs1]
            else:
                nebs1 = np.r_[[]]
                nebs2 = np.r_[nebs2]

            # overwrite segment ends
            if nebs1.size>0:
                print ('joining start %3d at' % i, endsw1[i], 'to start %3d at' % nebs1[0], endsw1[nebs1][0])
                rfile.xyz[i] = np.r_[[rfile.xyz[nebs1[0]][0]], rfile.xyz[i]]
                endsw1[i] = endsw1[nebs1][0]
            else:
                print ('joining start %3d at' % i, endsw1[i], 'to end   %3d at' % nebs2[0], endsw2[nebs2][0])
                rfile.xyz[i] = np.r_[[rfile.xyz[nebs2[0]][-1]], rfile.xyz[i]]
                endsw1[i] = endsw2[nebs2][0]
                                
        if nebs1.size+nebs2.size == 0:
            print ("WARNING, no neighbour found for start-point of segment %d" % i)
            terminating += 1
            
        if nebs1.size > 0:
            de11[i,nebs1] = 1
        if nebs2.size > 0:
            de12[i,nebs2] = 1


    for i in range(len(endsw2)):
          
        # first, check for junctions
        dis1,nebs1 = ktree1.query(endsw2[i], distance_upper_bound=djoin, k=8)
        dis2,nebs2 = ktree2.query(endsw2[i], distance_upper_bound=djoin, k=8)
        
        dis2  =  dis2[nebs2 != i]  # remove self interaction
        nebs2 = nebs2[nebs2 != i]
            
        nebs1 = nebs1[dis1<np.inf]
        nebs2 = nebs2[dis2<np.inf]
        
        # if no junction is found, check instead for the next closest link
        if nebs1.size + nebs2.size == 0:
            dis1,nebs1 = ktree1.query(endsw2[i], distance_upper_bound=dmax, k=1)
            dis2,nebs2 = ktree2.query(endsw2[i], distance_upper_bound=dmax, k=2)    
            
            dis2  =  dis2[1] # remove self interaction
            nebs2 = nebs2[1]
           
            if dis1 < dis2:
                nebs2 = np.r_[[]]
                nebs1 = np.r_[nebs1]
            else:
                nebs1 = np.r_[[]]
                nebs2 = np.r_[nebs2]


            # overwrite segment end
            if nebs1.size>0:
                #if i == nebs1[0]:
                #    raise ValueError("no nearest link found within cutoff radius %d." % dmax)

                print ('joining end   %3d at' % i, endsw2[i], 'to start %3d at' % nebs1[0], endsw1[nebs1][0])
                rfile.xyz[i] = np.r_[rfile.xyz[i], [rfile.xyz[nebs1[0]][0]]]
                endsw2[i] = endsw1[nebs1][0]
            else:
                #if i == nebs2[0]:
                #    raise ValueError("no nearest link found within cutoff radius %d." % dmax)

                print ('joining end   %3d at' % i, endsw2[i], 'to end   %3d at' % nebs2[0], endsw2[nebs2][0])
                rfile.xyz[i] = np.r_[rfile.xyz[i], [rfile.xyz[nebs2[0]][-1]]]
                endsw2[i] = endsw2[nebs2][0]

        if nebs1.size+nebs2.size == 0:
            print ("WARNING, no neighbour found for end-point of segment %d" % i)
            terminating += 1
                
        if nebs1.size > 0:
            de21[i,nebs1] = 1
        if nebs2.size > 0:
            de22[i,nebs2] = 1
    
    if terminating:
        successflag = False
        print ("WARNING: %d segments are still unconnected!" % terminating)
    else:
        print ("All segments connected.")
        successflag = True

    return de11, de12, de21, de22, endsw1, endsw2, successflag


def consistency_check (de11, de12, de21, de22):
    '''Consistency check; warns and returns 1 if there are unconnected segments.'''
    err = 0
    for i in range(len(de11)):
        if np.sum(de21[i])+np.sum(de22[i]) == 0:
            print ("Error, there are unconnected segments: %d" % i)
            err = 1
            
    for i in range(len(de11)):
        if np.sum(de11[i])+np.sum(de12[i]) == 0:
            print ("Error, there are unconnected segments: %d" % i)
            err = 1

    if not err:
        print ("Consistency check passed.\n")

    return err

