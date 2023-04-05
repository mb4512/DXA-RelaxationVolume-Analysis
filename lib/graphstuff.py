import numpy as np
import networkx as nx
from copy import deepcopy

def make_graph(rfile, edge_dict, endsw1, endsw2):
    g = nx.MultiDiGraph()

    for _edge in edge_dict.keys():
        _nodes,_bvec = edge_dict[_edge]
        _ni,_nj = _nodes
        g.add_edge(_ni, _nj, 
                   segid=_edge, seg=rfile.xyz[_edge], 
                   burgers=_bvec, 
                   ends=[endsw1[_edge],endsw2[_edge]])
    return g

def export_graph(g, path="graph.log"):
    A = nx.adjacency_matrix(g)
    Amat = np.array(A.todense(), dtype=int)
    print ("Exporting graph connectivity matrix to %s\n" % path)
    np.savetxt(path, Amat, fmt='%1d')

def segment_continuity(graph, rfile):
    # enforce continuity of segments across PBC
    for edge in graph.edges(data=True):
        _r = deepcopy(edge[2]['seg'])
        _dr = _r[1:]-_r[:-1]

        # convert to fractional coordinates and loop back into cell 
        _df = [rfile.cmati@(_dri-rfile.r0) for _dri in _dr]
        _df = [_dfi-np.round(_dfi) for _dfi in _df]

        # convert back to cartesian coordinates
        _dr = [rfile.r0 + rfile.cmat@_dfi for _dfi in _df]

        _r = edge[2]['ends'][0] + np.r_[np.zeros((1,3)), np.cumsum(_dr, axis=0)]
        edge[2]['seg'] = _r


def kirchhoff_check(graph): 
    '''Check if kirchhoff's law is violated.'''
    print ("Checking if Kirchhoff's law is violated.")
    viol=False
    for node in graph.nodes:
        e0 = list(graph.in_edges(node))
        e1 = list(graph.out_edges(node))
        
        #print ("node:", node) 
        bnet = np.r_[0.,0.,0.]
        done = []
        for _e in e0:
            if _e not in done:
                _edat = graph.get_edge_data(*_e)
                for _key in _edat:
                    #print ("\tin:", _edat[_key]['burgers'])
                    bnet += _edat[_key]['burgers']
            done += [_e]
            
        done = []
        for _e in e1:
            if _e not in done:            
                _edat = graph.get_edge_data(*_e)
                for _key in _edat:
                    #print ("\tout:", -_edat[_key]['burgers'])
                    bnet -= _edat[_key]['burgers']
            done += [_e]
        
        if (np.abs(bnet)>1e-6).any():
            #print ([graph.get_edge_data(*_e0) for _e0 in e0])
            #print ([graph.get_edge_data(*_e1) for _e1 in e1])
            print ('node %3d, net burgers:' % node, bnet)
            viol=True

    if viol:
        print ("Kirchhoff's law is violated. The network has inconsistent nodes.\n") 
        return 1
    else:
        print ("Kirchhoff's law is met.\n")
        return 0

def recursive_link(_subgraph, processed_edges, processed_nodes, scheduled_nodes):
    '''Loop over all edges in the graph and unwrap the graph.'''    
    maxrecursions = 10000
    
    for _recursions in range(maxrecursions):
        add_to_schedule = []

        for _newnode in scheduled_nodes:
            _pos,_node = _newnode
            if _node in processed_nodes:
                continue

            _sedges = np.array(_subgraph.edges)

            # select outgoing edges and move them to the anchor point
            _outgoing = np.where(_sedges[:,0] == _node)[0]
            for _s in _sedges[_outgoing]:   
                if tuple(_s) in processed_edges:
                    continue

                _edgedata = _subgraph.get_edge_data(*_s)
                _edgedata['seg'] = _edgedata['seg'] - _edgedata['seg'][0] + _pos
                processed_edges += [tuple(_s)]

                # add nodes terminating at outgoing edges to schedule
                add_to_schedule += [[_edgedata['seg'][-1], _s[1]]]

            # select ingoing edges and move them to the anchor point
            _ingoing = np.where(_sedges[:,1] == _node)[0]
            for _s in _sedges[_ingoing]:
                if tuple(_s) in processed_edges:
                    continue

                _edgedata = _subgraph.get_edge_data(*_s)
                _edgedata['seg'] = _edgedata['seg'] - _edgedata['seg'][-1] + _pos
                processed_edges += [tuple(_s)] 

                # add nodes beginning at incoming edges to schedule
                add_to_schedule += [[_edgedata['seg'][0], _s[0]]]

            processed_nodes += [_node]

        scheduled_nodes += add_to_schedule

        if add_to_schedule == []:
            _lpe,_lse = len(processed_edges), len(_subgraph.edges)
            if _lpe != _lse:
                print ("Warning: schedule empty but only %d out of %d edges processed." % (_lpe, _lse))
            _lpn,_lsn = len(processed_nodes), len(_subgraph.nodes)
            if _lpn != _lsn:
                print ("Warning: schedule empty but only %d out of %d nodes processed." % (_lpn, _lsn))

            break
    
    if add_to_schedule != []:
        print ("Error: did not finish within %d recursions." % maxrecursions)
        return 1
    else:
        return 0


def relink_graph(graph):
    '''Unwraps all subgraphs in the graph.'''
    gcopy = deepcopy(graph)
    
    for _i,_subgraph in enumerate(gcopy):
        processed_edges = []
        processed_nodes = []
        scheduled_nodes = []

        #_outedge = list(_subgraph.edges)[min(38, len(list(_subgraph.edges))-1)]
        _outedge = list(_subgraph.edges)[0]

        anchor_pos = _subgraph.get_edge_data(*_outedge)['seg'][0]
        anchor_node = _outedge[0]
        scheduled_nodes += [[anchor_pos, anchor_node]]
        processed_edges += [_outedge]

        ''' 
        if _i == 28:
            # these are all the edges
            _edges = list(_subgraph.edges)
            for _edge in _edges: 
                print (_edge, _subgraph.get_edge_data(*_edge)['seg'])
            return 0
        '''

        _res = recursive_link(_subgraph, processed_edges, processed_nodes, scheduled_nodes)
        if _res == 1:
            print ("Warning! Failed at subgraph %d." % _i)
    return gcopy





if 0:
    def relaxationvolume(sgraph, alattice, omega0, offset=[0,0,0], verbose=1):
        '''Compute the relaxation volume of all closed networks.'''

        offset = np.r_[offset]
        
        omegalist = []
        for _i,_subgraph in enumerate(sgraph):
            omega = 0

            for edge in _subgraph.edges(data=True):
                _b = edge[2]['burgers']
                _r = edge[2]['seg']
                _dr = _r[1:]-_r[:-1]

                omega += alattice*np.sum([.5*np.dot(_b, _q) for _q in np.cross(_r[:-1]-offset, _dr, axis=1)])
            omegalist += [omega]
            if verbose:
                print ('Relaxation volume of subgraph %4d: %14.8f atomic volumes' % (_i, omega/omega0))
           
        omegalist = np.r_[omegalist]
        omegatot = np.sum(omegalist)
        if verbose:
            print ('Total relaxation volume: %14.8f atomic volumes' % (omegatot/omega0))

        return omegatot/omega0, omegalist/omega0

def linelength(sgraph, verbose=1):
    '''Compute the dislocation line lengths of all segments and classify by Burgers vector norm.'''

    perimeterdict = {}
    perimetertot = {}
    for _i,_subgraph in enumerate(sgraph):
        perimeter = 0

        for edge in _subgraph.edges(data=True):
            _bnorm = np.linalg.norm(edge[2]['burgers'])
            _r = edge[2]['seg']
            perimeter += np.sum([np.linalg.norm(_r[_j+1]-_r[_j]) for _j in range(len(_r)-1)]) 

        if _bnorm not in perimeterdict:
            perimeterdict[_bnorm] = []
        perimeterdict[_bnorm] += [perimeter]

        if verbose:
            print ('Perimeter length of subgraph %4d: %14.8f Angstrom' % (_i, perimeter))
    
    for _key in perimeterdict:   
        perimeterdict[_key] = np.r_[perimeterdict[_key]]
        perimetertot[_key] = np.sum(perimeterdict[_key])

    perimeterall = 0.0
    for _key in perimetertot:
        perimeterall += perimetertot[_key] 

    if verbose:
        print ('Total perimeter length: %14.8f Angstrom' % (perimeterall))

    return perimeterall, perimetertot, perimeterdict



def relaxationvolume(sgraph, alattice, omega0, offset=[0,0,0], verbose=1):
    '''Compute the relaxation volume of all closed networks.'''

    offset = np.r_[offset]
    
    omegalist = []
    for _i,_subgraph in enumerate(sgraph):
        omega = 0

        for edge in _subgraph.edges(data=True):
            _b = edge[2]['burgers']
            _r = edge[2]['seg']
            omega += alattice*np.sum([.5*np.dot(_b, np.cross(_r[_j], _r[_j+1])) for _j in range(len(_r)-1)])

        omegalist += [omega]
        if verbose:
            print ('Relaxation volume of subgraph %4d: %14.8f atomic volumes' % (_i, omega/omega0))
       
    omegalist = np.r_[omegalist]
    omegatot = np.sum(omegalist)
    if verbose:
        print ('Total relaxation volume: %14.8f atomic volumes' % (omegatot/omega0))

    return omegatot/omega0, omegalist/omega0

def relaxationvolumetensor(sgraph, alattice, omega0, offset=[0,0,0], verbose=1):
    '''Compute the relaxation volume tensor of all closed networks.'''

    print ('Computing relaxation volume tensors of in atomic volumes.')
    offset = np.r_[offset]
    omegaijlist = []
    for _i,_subgraph in enumerate(sgraph):
        omegaij = np.zeros((3,3))

        for edge in _subgraph.edges(data=True):
            _b = edge[2]['burgers']
            _r = edge[2]['seg']

            _outer = alattice*np.sum([.5*np.outer(_b, np.cross(_r[_j], _r[_j+1])) for _j in range(len(_r)-1)], axis=0)
            omegaij += .5*(_outer + _outer.T)

        omegaijlist += [omegaij]
        if verbose:
            print ('Subgraph %4d: ' % _i)
            print ("%12.4f %12.4f %12.4f" % tuple(omegaij[0]/omega0))
            print ("%12.4f %12.4f %12.4f" % tuple(omegaij[1]/omega0))
            print ("%12.4f %12.4f %12.4f" % tuple(omegaij[2]/omega0))
            print ()
 
    omegaijlist = np.r_[omegaijlist]
    omegaijtot = np.sum(omegaijlist, axis=0)

    if verbose:
        print ('Total relaxation volume tensor in atomic volumes: ') 
        print ("%12.4f %12.4f %12.4f" % tuple(omegaijtot[0]/omega0))
        print ("%12.4f %12.4f %12.4f" % tuple(omegaijtot[1]/omega0))
        print ("%12.4f %12.4f %12.4f" % tuple(omegaijtot[2]/omega0))
        print ()
 
    return omegaijtot/omega0, omegaijlist/omega0

def pbc_volume_correction(sgraph, alattice, omega0, rfile, verbose=True):
    ''''''

    # loop over all nodes and check if they are split by PBC. save their id and position
    print ("Searching for threading dislocations by checking for dangling nodes.")
    dangling_nodes = []
    subgraph_ids = []


    print ("\nLooking for dangling nodes.")   
    found = False
    dangling_nodes = []
    for _sid,_subgraph in enumerate(sgraph):
        _sedges = np.array(_subgraph.edges)
        for _node in _subgraph.nodes:
            
            # first check outgoing edges for this node
            _outgoing = np.where(_sedges[:,0] == _node)[0]
            _roots_out = [_subgraph.get_edge_data(*_s)['seg'][0] for _s in _sedges[_outgoing]]
            
             # then check ingoing edges for this node
            _ingoing = np.where(_sedges[:,1] == _node)[0]
            _roots_in = [_subgraph.get_edge_data(*_s)['seg'][-1] for _s in _sedges[_ingoing]]
            
            # if the edges attached to the node have different roots, it is a dangling node
            _roots  = np.array(_roots_out + _roots_in)
            _uroots = np.unique(np.round(_roots, 6), axis=0) # it is annoying but we need to round since checking unique of floats
            _rootdist = np.linalg.norm(_uroots-_roots[0], axis=1)

            # we only care about the nodes that are actually dangling 
            _bool = (_rootdist > 1e-3)

            # store dangling node, unique pbc-displaced node positions, and the corresponding pbc image vector
            if np.sum(_bool) > 0:
                if not found:
                    print ("graph id, node id, root node pos, pbc-displaced pos, node-closer-to-corner pos, difference vector in frac.coordinates")
                    found = True

                #print (_uroots, _bool)

                # we store information on the dangling node in a dictionary for better access 
                _node_dict = {}
                _node_dict["subgraph_id"] = _sid
                _node_dict["node_id"] = _node
                _node_dict["root_node"] = _roots[0]
                _node_dict["pbc_nodes"] = _uroots[_bool]

                # difference vector
                _dvecs = _uroots[_bool] - _roots[0]
                _dist = np.linalg.norm(_dvecs, axis=1)

                # convert image vector to fractional coordinates
                _fvecs = [rfile.cmati@_dd for _dd in _dvecs]
                _fvecs = np.round(_fvecs).astype(int)

                '''
                # sign convention
                for _fvec in _fvecs:
                    for _fi in range(3):
                        if _fvec[2-_fi] < 0:
                            _fvec *= -1
                '''

                # avoid -0.0...
                for _fvec in _fvecs:
                    _fvec[np.abs(_fvec) < 1e-9] = 0.0

                # return image vector to cartesian coordinates 
                _cvecs = [rfile.cmat@_fvec for _fvec in _fvecs]
                for _cvec in _cvecs:
                    _cvec[np.abs(_cvec) < 1e-9] = 0.0

                _node_dict["frac_vecs"] = _fvecs
                _node_dict["pbc_vecs"] = _cvecs

                # determine node lying closer to the [111] point
                _node_dict["corner_pos"] = rfile.cmat@np.r_[1,1,1] + rfile.r0
                _node_dict["connecting_node"] = np.zeros_like(_node_dict["pbc_nodes"])
                for _pi,_pos2 in enumerate(_node_dict["pbc_nodes"]):
                    _pos1 = _node_dict["root_node"]
                    _d1 = np.linalg.norm(_node_dict["corner_pos"]-_pos1) 
                    _d2 = np.linalg.norm(_node_dict["corner_pos"]-_pos2) 
                    if _d1 < _d2:
                        _node_dict["connecting_node"][_pi] = _pos1
                    else:
                        _node_dict["connecting_node"][_pi] = _pos2
                        _node_dict["frac_vecs"][_pi] *= -1
                        _node_dict["pbc_vecs"][_pi]  *= -1

                for _pi,_pos2 in enumerate(_node_dict["pbc_nodes"]):
                    _pos1 = _node_dict["root_node"]
                    _fvec = _node_dict["frac_vecs"][_pi]
                    _cpos =  _node_dict["connecting_node"][_pi]
                    print ("\t%5d %5d [%8.3f %8.3f %8.3f]  [%8.3f %8.3f %8.3f]  [%8.3f %8.3f %8.3f]  [%2d %2d %2d]" % (_node_dict["subgraph_id"], _node_dict["node_id"], 
                                                        _pos1[0], _pos1[1], _pos1[2], _pos2[0], _pos2[1], _pos2[2], 
                                                        _cpos[0], _cpos[1], _cpos[2], _fvec[0], _fvec[1], _fvec[2]))


                dangling_nodes += [_node_dict]

    # stop routine if no broken nodes are found (no threading dislocations) 
    if dangling_nodes == []:
        print ("No threading dislocations found.\n")
        return False 

    # determine closure condition of the dangling node
    print ("Determining (outgoing) Burgers vector needed to close the node according to Kirchhoff's law.")
    print ("graph id, node id, connecting node index, burgers vector")
    for _node_dict in dangling_nodes:
        _node = _node_dict["node_id"]
        _subgraph = sgraph[_node_dict["subgraph_id"]]
        _sedges = np.array(_subgraph.edges)

        _node_dict["bclosure"] = np.zeros_like(_node_dict["connecting_node"])

        for _cpi,_cpos in enumerate(_node_dict["connecting_node"]):
            # fetch outgoing edges of this node and keep the ones that are unbroken
            _outgoing = np.where(_sedges[:,0] == _node)[0]
            _roots_out = [_subgraph.get_edge_data(*_s)['seg'][0] for _s in _sedges[_outgoing]]
            _bout = []
            if _roots_out:
                _outedges = _sedges[_outgoing][np.linalg.norm(_roots_out - _cpos, axis=1)<1e-3]
                _bout += [_subgraph.get_edge_data(*_s)['burgers'] for _s in _outedges]
                
            # fetch ingoing edges of this node and keep the ones that are unbroken
            _ingoing = np.where(_sedges[:,1] == _node)[0]
            _roots_in = [_subgraph.get_edge_data(*_s)['seg'][-1] for _s in _sedges[_ingoing]]
            _bin = []
            if _roots_in:
                _inedges = _sedges[_ingoing][np.linalg.norm(_roots_in - _cpos, axis=1)<1e-3]
                _bin += [_subgraph.get_edge_data(*_s)['burgers'] for _s in _inedges]
     
            # compute burgers vector needed to close 
            _bclosure = np.zeros(3) 
            if _bin != []:
                _bclosure += np.sum(_bin, axis=0)
            if _bout != []:
                _bclosure -= np.sum(_bout, axis=0)

            _node_dict["bclosure"][_cpi] = _bclosure
            print ("\t%5d %5d %2d [%8.3f %8.3f %8.3f]" % (_node_dict["subgraph_id"], _node, _cpi, _bclosure[0], _bclosure[1], _bclosure[2])) 


    # loop over all broken nodes and compute relaxation volume correction 
    print ("Computing periodic closure correction to the relaxation volume.")
    #print ("bvec, dangling node pos, pbc image vector, root node pos")
    _omega = 0
    _bnet = np.zeros(3) 
    for _ndi,_node_dict in enumerate(dangling_nodes):
        _node = _node_dict["node_id"]
        _subgraph = sgraph[_node_dict["subgraph_id"]]
        _sedges = np.array(_subgraph.edges)
        _corner_pos = _node_dict["corner_pos"]

        for _cpi,_cpos in enumerate(_node_dict["connecting_node"]):

            _bvec = _node_dict["bclosure"][_cpi]
            _cvec = _node_dict["pbc_vecs"][_cpi]
            _omega += .5*np.dot(_bvec, np.cross(_cvec, -_corner_pos+_cpos))
            _bnet += _bvec 

            #_term =  -.5*np.dot(_bvec, np.cross(_cvec, -_corner_pos+_cpos))
            #print ("\t[%8.3f %8.3f %8.3f]  [%8.3f %8.3f %8.3f]  [%8.3f %8.3f %8.3f]  [%8.3f %8.3f %8.3f] %8.3f" % (
            #                                    _bvec[0], _bvec[1], _bvec[2], _cpos[0], _cpos[1], _cpos[2], _cvec[0], _cvec[1], _cvec[2],
            #                                    _corner_pos[0], _corner_pos[1], _corner_pos[2], _term))

    if not (_bnet == 0).all():
        print ("Warning: net Burgers vector in corner node is not zero. Net b:", _bnet) 
        print ()
        #return False


    # loop over all broken nodes and compute relaxation volume tensor correction 
    print ("Computing periodic closure correction to the relaxation volume tensor.")
    _omegaij = np.zeros((3,3))
    for _ndi,_node_dict in enumerate(dangling_nodes):
        _node = _node_dict["node_id"]
        _subgraph = sgraph[_node_dict["subgraph_id"]]
        _sedges = np.array(_subgraph.edges)
        _corner_pos = _node_dict["corner_pos"]

        for _cpi,_cpos in enumerate(_node_dict["connecting_node"]):

            _bvec = _node_dict["bclosure"][_cpi]
            _cvec = _node_dict["pbc_vecs"][_cpi]
            _outer = .5*np.outer(_bvec, np.cross(_cvec, -_corner_pos+_cpos))
            _omegaij += .5*(_outer + _outer.T) 


    print ("Relaxation volume correction: %14.8f atomic volumes" % (alattice*_omega/omega0))
    print ()

    if verbose:
        print ('Periodic relaxation volume tensor correction in atomic volumes: ') 
        print ("%12.4f %12.4f %12.4f" % tuple(alattice*_omegaij[0]/omega0))
        print ("%12.4f %12.4f %12.4f" % tuple(alattice*_omegaij[1]/omega0))
        print ("%12.4f %12.4f %12.4f" % tuple(alattice*_omegaij[2]/omega0))
        print ()
 
    return alattice*_omega/omega0, alattice*_omegaij/omega0


