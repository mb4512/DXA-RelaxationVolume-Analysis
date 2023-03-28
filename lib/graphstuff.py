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


def pbcdist(x, y, rfile):
    '''Compute distance between points x and y in the periodic orthogonal box stored in rfile.'''
    _x = x - rfile.cell*np.floor(x/rfile.cell)
    _y = y - rfile.cell*np.floor(y/rfile.cell)
    
    _dr = _x-_y
    _dr = _dr - rfile.cell*np.trunc(_dr/(.5*rfile.cell))
    return np.linalg.norm(_dr)

def pbcdistr(dr, rfile):
    '''Wrap distance vector dr in the periodic orthogonal box stored in rfile.'''
    _dr = deepcopy(dr)
    _dr = _dr - rfile.cell*np.floor(_dr/rfile.cell)
    _dr = _dr - rfile.cell*np.trunc(_dr/(.5*rfile.cell))
    return np.linalg.norm(_dr)

def pbcdistvec(dr, rfile):
    '''Wrap distance vector dr in the periodic orthogonal box stored in rfile.'''
    _dr = deepcopy(dr)
    _dr = _dr - rfile.cell*np.floor(_dr/rfile.cell)
    _dr = _dr - rfile.cell*np.trunc(_dr/(.5*rfile.cell))
    return _dr


def flipGraph(graph, rfile):
    ''' Smoothen the graph until all nodes of degree 2 have one incident and one outgoing edge. '''

    print ("Smoothing the graph until all nodes of degree 2 have one incident and one outgoing edge.")
    print ("Networks with inconsistent linesense may chance character to become consistent.")
    g = deepcopy(graph)

    if not g.is_directed():
        print ('Error: graph is undirected.')
        return g
    
    if min(dict(g.degree).values()) < 2:
        print ('Error: graph contains nodes with fewer than 2 edges.')
        return g
    
    # check for convergence by monitoring the number of nodes in the graph
    nnodes0 = np.sum(list(dict(g.degree).values()))
    
    for _iters in range(1000): # max iterations
        print ("Iteration number %d" % _iters)
        g0 = g.copy() #<- simply changing g itself would cause error `dictionary changed size during iteration` 
        inode = -1
        for node, degree in g.degree():
            inode += 1
            
            if degree==2:

                e0 = list(g.in_edges(node))
                e1 = list(g.out_edges(node))

                n0 = len(e0)
                n1 = len(e1)
    
                # for balanced nodes
                if n0==1 and n1==1:
                    edge0 = g.get_edge_data(*e0[0])[0]
                    edge1 = g.get_edge_data(*e1[0])[0]

                    # skip self-loops, they are already fine
                    if edge0['segid'] == edge1['segid']:
                        continue

                    # check consistency with burgers vector
                    kirchhoff = edge0['burgers'] - edge1['burgers']
                    if np.sum(np.abs(kirchhoff)) > 1e-3:
                        print ('Error: Inconsistent Kirchhoff rule at node %d!' % node)
                        return 1

                    a0,b0 = e0[0]
                    a1,b1 = e1[0]

                    q0 = a0 if a0!=node else b0
                    q1 = a1 if a1!=node else b1

                    # figure out in which order to attach segments by computing
                    # the distance between their endpoints
                    start1,end1 = edge0['ends']
                    start2,end2 = edge1['ends']
                    
                    d1 = pbcdist(end1, start2, rfile)
                    d2 = pbcdist(end2, start1, rfile)
                    
                    if d1 < d2:
                        newseg  = np.r_[edge0['seg'], edge1['seg']]
                        newends = [start1, end2]
                        print ('+- joining at:', end1, start2)
                    else:
                        newseg = np.r_[edge1['seg'], edge0['seg']]
                        newends = [start2, end1] 
                        print ('+- joining at:', end2, start1)
                    
                    newids = edge0['segid'] + edge1['segid']
                    
                    # concatenate node
                    g0.remove_node(node)
                    g0.add_edge(q0, q1, burgers=edge0['burgers'], ends=newends, seg=newseg, segid=newids)
                    
                    _dr = newseg[1:]-newseg[:-1]
                    _lengths = [pbcdistr(dr, rfile) for dr in _dr]
                    if np.max(_lengths) > 10:
                        print ('^^^ Error: broken chain! See coordinates below vvv')
                        print ('node %d' % inode)
                        print (newseg)
                        print ()
                        return 1

                # two incoming edges
                if n0==2:
                    # flip and merge one of them
                    edge0 = g.get_edge_data(*e0[0])[0] 
                    edge1 = g.get_edge_data(*e0[1])
                    if 1 in edge1:
                        edge1 = edge1[1]
                    else:
                        edge1 = edge1[0]

                    # check consistency with burgers vector, assuming incident edge nr.2 is flipped
                    kirchhoff = edge0['burgers'] + edge1['burgers']
                    if np.sum(np.abs(kirchhoff)) > 1e-3:
                        print ('Error: Inconsistent Kirchhoff rule at node %d!' % node)
                        return 1

                    a0,b0 = e0[0]
                    a1,b1 = e0[1]  

                    q0 = a0 if a0!=node else b0
                    q1 = a1 if a1!=node else b1
                    
                    # figure out in which order to attach segments by computing
                    # the distance between their endpoints
                    start1,end1 = edge0['ends']
                    start2,end2 = edge1['ends']
                    
                    d1 = pbcdist(end1, end2, rfile)
                    d2 = pbcdist(start1, start2, rfile)
                    
                    if d1 < d2:
                        newseg  = np.r_[edge0['seg'], np.flip(edge1['seg'], axis=0)]
                        newends = [start1, start2]
                        print ('++ joining at:', end1, end2)
                    else:
                        newseg = np.r_[np.flip(edge1['seg'], axis=0), edge0['seg']]
                        newends = [end2, end1] 
                        print ('++ joining at:', start1, start2)
                    
                    newids = edge0['segid'] + edge1['segid']
                    
                    # concatenate node
                    g0.remove_node(node)  
                    g0.add_edge(q0, q1, burgers=edge0['burgers'], ends=newends, seg=newseg, segid=newids)

                    _dr = newseg[1:]-newseg[:-1]
                    _lengths = [pbcdistr(dr, rfile) for dr in _dr]
                    if np.max(_lengths) > 10:
                        print ('^^^ Error: broken chain! See coordinates below vvv')
                        print (newseg)
                        print ()
                        return 1

                # two outgoing edges
                if n1==2:
                    # flip and merge one of them
                    edge0 = g.get_edge_data(*e1[0])[0]
                    edge1 = g.get_edge_data(*e1[1])
                    if 1 in edge1:
                        edge1 = edge1[1]
                    else:
                        edge1 = edge1[0]

                    # check consistency with burgers vector, assuming incident edge nr.2 is flipped
                    kirchhoff = edge0['burgers'] + edge1['burgers']
                    if np.sum(np.abs(kirchhoff)) > 1e-3:
                        print ('Error: Inconsistent Kirchhoff rule at node %d!' % node)

                    a0,b0 = e1[0]
                    a1,b1 = e1[1]  

                    q0 = a0 if a0!=node else b0
                    q1 = a1 if a1!=node else b1

                    # figure out in which order to attach segments by computing
                    # the distance between their endpoints
                    start1,end1 = edge0['ends']
                    start2,end2 = edge1['ends']
                    
                    d1 = pbcdist(start1, start2, rfile)
                    d2 = pbcdist(end1, end2, rfile)
                    
                    if d1 < d2:
                        newseg  = np.r_[np.flip(edge0['seg'], axis=0), edge1['seg']]
                        newends = [end1, end2]
                        print ('-- joining at:', start1, start2)
                    else:
                        newseg = np.r_[edge1['seg'], np.flip(edge0['seg'], axis=0)]
                        newends = [start2, start1] 
                        print ('-- joining at:', end1, end2)
                    
                    newids = edge0['segid'] + edge1['segid']

                    # concatenate node
                    g0.remove_node(node)  
                    g0.add_edge(q0, q1, burgers=edge1['burgers'], ends=newends, seg=newseg, segid=newids)
                    
                    _dr = newseg[1:]-newseg[:-1]
                    _lengths = [pbcdistr(dr, rfile) for dr in _dr]
                    if np.max(_lengths) > 10:
                        print ('^^^ Error: broken chain! See coordinates below vvv')
                        print (newseg)
                        print ()
                        return 1                        
                else:
                    pass
            g = g0
            
        nnodes = np.sum(list(dict(g.degree).values()))
        
        if nnodes==nnodes0:
            print ('Graph converged after %d iterations.' % (_iters+1))
            break
        nnodes0 = nnodes
        
    
    if nnodes!=nnodes0:
        print ('Graph did not converge after %d iterations.' % (_iters+1))
        return 1
        
    return g

def segment_continuity(graph, rfile):
    # enforce continuity of segments across PBC
    for edge in graph.edges(data=True):
        _r = deepcopy(edge[2]['seg'])
        _q = _r[1:]-_r[:-1]
        _q = np.sign(_q)*(np.abs(_q)%rfile.cell) # valid beyond min. image convention
        _q = _q - rfile.cell*np.trunc(_q/(.5*rfile.cell))
        
        _r = edge[2]['ends'][0] + np.r_[[np.r_[0,0,0]], np.cumsum(_q, axis=0)]
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

        _outedge = list(_subgraph.edges)[0]

        anchor_pos = _subgraph.get_edge_data(*_outedge)['seg'][0]
        anchor_node = _outedge[0]
        scheduled_nodes += [[anchor_pos, anchor_node]]
        processed_edges += [_outedge]
        
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
    for _sid,_subgraph in enumerate(sgraph):
        _sedges = np.array(_subgraph.edges)
        for _node in _subgraph.nodes:
            
            # first check outgoing edges
            _outgoing = np.where(_sedges[:,0] == _node)[0]
            _roots_out = [_subgraph.get_edge_data(*_s)['seg'][0] for _s in _sedges[_outgoing]]
            
             # then check ingoing edges
            _ingoing = np.where(_sedges[:,1] == _node)[0]
            _roots_in = [_subgraph.get_edge_data(*_s)['seg'][-1] for _s in _sedges[_ingoing]]
            
            # if the edges attached to the node have differing roots, it is a dangling node
            _roots  = np.array(_roots_out + _roots_in)
            _rootdist = np.linalg.norm(_roots-_roots[0], axis=1)
            if np.sum(_rootdist) > 1e-3:
                
                # store dangling node and unique pbc-displaced node positions
                dangling_xyz = np.vstack(list({tuple(row) for row in _roots}))        
                dangling_nodes += [[_node, dangling_xyz]]
                subgraph_ids += [_sid]
 
    # stop routine if no broken nodes are found (no threading dislocations) 
    if dangling_nodes == []:
        print ("No threading dislocations found.\n")
        return False 

    # for every dangling node, fetch the image vector
    cvecs = []
    cmatrix = np.diag(rfile.cell)
    if 1:
        for _d in dangling_nodes:
            _diff = _d[1][1]-_d[1][0]
            _cvec = np.sum(np.sign(_diff)*cmatrix[np.abs(_diff)>1.0], axis=0)

            # convention
            for _q in range(3):
                if _cvec[_q] < 0:
                    _cvec *= -1

            # avoid -0.0...
            _cvec[np.abs(_cvec) < 1e-6] = 0.0
            cvecs += [_cvec]
 
    if 0:
        for _d in dangling_nodes:
            _diff = _d[1][1]-_d[1][0]
            _cvec = np.sum(np.sign(_diff)*cmatrix[np.abs(_diff)>1.0], axis=0)
            if _cvec[0] < 0:
                _cvec *= -1
            cvecs += [_cvec]
    
    for _i in range(len(cvecs)): 
        _pos1 =  dangling_nodes[_i][1][0]
        _pos2 =  dangling_nodes[_i][1][1]
        _vec = cvecs[_i]
        print ("\t%5d %5d [%8.3f %8.3f %8.3f] [%8.3f %8.3f %8.3f]  [%8.3f %8.3f %8.3f]" % (subgraph_ids[_i], dangling_nodes[_i][0], 
                                                                        _pos1[0], _pos1[1], _pos1[2], _pos2[0], _pos2[1], _pos2[2], _vec[0], _vec[1], _vec[2]))
    print ()

       
    # of the two dangling node images, keep the one with largest value projected along the pbc direction 
    for _i,_d in enumerate(dangling_nodes):
        _proj = [np.dot(_npos, cvecs[_i]) for _npos in _d[1]]
        _d[1] = _d[1][np.argmax(_proj)]

    print ("Found the following dangling nodes with corresponding PBC image vector:")
    print ("graph id, node id, position, image vector")
    for _i in range(len(cvecs)): 
        _pos =  dangling_nodes[_i][1]
        _vec = cvecs[_i]
        print ("\t%5d %5d [%8.3f %8.3f %8.3f]  [%8.3f %8.3f %8.3f]" % (subgraph_ids[_i], dangling_nodes[_i][0], _pos[0], _pos[1], _pos[2], _vec[0], _vec[1], _vec[2]))
    print ()

    # group nodes with unique cvecs together as they are bundled separately 
    cvecs_grouping = np.unique(cvecs, axis=0, return_inverse=True)

    # build dictionary containing the dangling node id and corresponding outgoing burgers vector needed for closure 
    print ("Determining (outgoing) Burgers vector needed to close the node according to Kirchhoff's law.")
    print ("grpah id, node id, burgers vector")
    bclosures = {}
    for _i,_dang in enumerate(dangling_nodes):
        _node,_pos = _dang
        _subgraph = sgraph[subgraph_ids[_i]]
        _sedges = np.array(_subgraph.edges)

        # fetch outgoing edges and keep the ones that emerge specifically from that node image
        _outgoing = np.where(_sedges[:,0] == _node)[0]
        _roots_out = [_subgraph.get_edge_data(*_s)['seg'][0] for _s in _sedges[_outgoing]]
        if _roots_out:
            _outedges = _sedges[_outgoing][np.linalg.norm(_roots_out - _pos, axis=1)<1e-3]
            _bout = [_subgraph.get_edge_data(*_s)['burgers'] for _s in _outedges]
        else:
            _bout = []
            
        # fetch ingoing edges and keep the ones that emerge specifically from that node image
        _ingoing = np.where(_sedges[:,1] == _node)[0]
        _roots_in = [_subgraph.get_edge_data(*_s)['seg'][-1] for _s in _sedges[_ingoing]]
        if _roots_in:
            _inedges = _sedges[_ingoing][np.linalg.norm(_roots_in - _pos, axis=1)<1e-3]
            _bin  = [_subgraph.get_edge_data(*_s)['burgers'] for _s in _inedges]
        else:
            _bin = []
       
        # compute burgers vector needed to close 
        _bclosure = np.r_[[0.,0.,0.]]
        if _bin != []:
            _bclosure += np.sum(_bin, axis=0)
        if _bout != []:
            _bclosure -= np.sum(_bout, axis=0)
        bclosures[_node] = _bclosure

        print ("\t%5d %5d [%8.3f %8.3f %8.3f]" % (subgraph_ids[_i], _node, _bclosure[0], _bclosure[1], _bclosure[2])) 
    print ()

    
    # loop over all broken nodes and compute relaxation volume correction 
    print ("Computing periodic closure correction to the relaxation volume.")
    _omega = 0
    for _ci,_cvec in enumerate(cvecs_grouping[0]):
        # fetch nodes with that cell vector 
        _nodes = [_d for _i,_d in enumerate(dangling_nodes) if (cvecs_grouping[1]==_ci)[_i]]
        _bvecs = [bclosures[_n[0]] for _n in _nodes]
        
        for _i in range(1, len(_nodes)):
            _omega += -.5*np.dot(_bvecs[_i], np.cross(_cvec, _nodes[_i][1]-_nodes[0][1]))
            _bnet = _bvecs[0] + np.sum(_bvecs[1:], axis=0)
            if not (_bnet == 0).all():
                print ("Error: net Burgers vector in grouping %3d is not zero!" % _ci, " net b:", _bnet) 
                print ()
                return False

    # loop over all broken nodes and compute relaxation volume tensor correction 
    print ("Computing periodic closure correction to the relaxation volume tensor.")
    _omegaij = np.zeros((3,3))
    for _ci,_cvec in enumerate(cvecs_grouping[0]):
        # fetch nodes with that cell vector 
        _nodes = [_d for _i,_d in enumerate(dangling_nodes) if (cvecs_grouping[1]==_ci)[_i]]
        _bvecs = [bclosures[_n[0]] for _n in _nodes]
        
        for _i in range(1, len(_nodes)):
            _outer = -.5*np.outer(_bvecs[_i], np.cross(_cvec, _nodes[_i][1]-_nodes[0][1]))
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


