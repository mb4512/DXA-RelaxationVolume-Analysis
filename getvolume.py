#!/usr/local/bin/python3
import sys, glob

from lib.readfile import *
from lib.dxa_process import *
from lib.graphstuff import *

def main():

    fpath = sys.argv[1]
    if len(sys.argv)>2:
        dpath = sys.argv[2]
    else:
        dpath = None

    # import dxa data file    
    rfile = ReadDump(fpath)
    rfile.load(dxa=True)

    # connect terminating segments
    for i in range(3):
        de11,de12,de21,de22,endsw1,endsw2,successflag = link_network(rfile, dmax=15.)
        #if successflag is True:
        #    break
 
    if consistency_check (de11, de12, de21, de22) == 1:
        return 1

    # build a register of dislocation nodes (and junctions)
    nodes_reg = nodes_register(rfile, de11, de12, de21, de22)
    if nodes_reg == 1:
        print ("Failed to build a consistent node register.")
        return 1

    # construct dictionaries containing edge and node information 
    edge_dict, node_dict = make_dictionaries(nodes_reg)

    # construct graph representation of dislocation network
    graph = make_graph(rfile, edge_dict, endsw1, endsw2)
 
    ''' 
    export_graph(graph, path="graph.log")
    '''

    if kirchhoff_check(graph) == 1:
        return 1

    # ensure segments are continuous across PBC
    segment_continuity(graph, rfile)

    # unwrap segments
    sgraph = [nx.MultiDiGraph(deepcopy(graph.subgraph(c))) for c in nx.weakly_connected_components(graph)]
    sshifted = relink_graph(sgraph)

    # at this point, check which subgraphs are large enough to potentially be extended
    print ("Looking for extended networks.")
    large_network_dict = {}
    for _sid,_subgraph in enumerate(sshifted):
        _sgraphxyz = np.vstack([edge[2]['seg'] for edge in _subgraph.edges(data=True)])
    
        # get x,y,z spans of subgraph
        xmin = np.min(_sgraphxyz, axis=0) 
        xmax = np.max(_sgraphxyz, axis=0) 
        xdel = xmax-xmin

        # convert to fractional coordinates
        fdel = rfile.cmati@xdel
        fextended = fdel>=1 

        if (fextended == True).any():
            large_network_dict[_sid] = fextended
            print ("subgraph %d is larger than box dimensions along x, y, z:" % _sid, fextended)

            '''
            # for this subgraph, wrap the xyz coordinates of each segment back into the box along the extended dimension 
            for edge in _subgraph.edges(data=True):
                _wrapped = np.array([rfile.pbcwrap(_x) for _x in edge[2]['seg']])
                for _bi,_extended in enumerate(large_network_dict[_sid]):
                    if _extended:
                        edge[2]['seg'][:,_bi] = _wrapped[:,_bi]
            '''

    if large_network_dict == {}:
        print ("No networks found that are larger than box dimensions. No extended networks present.")
   
    #extended_dict = large_network_dict

    #'''
    # if networks exist that are larger than any dimension of the box, check if they are extended 
    extended_dict = {}
    for _sid in large_network_dict:

        print ("\nLooking for dangling nodes in subgraph %d." % _sid)   
        _subgraph = sshifted[_sid]
        _sedges = np.array(_subgraph.edges)

        # loop over all nodes of the subgraph and check if they are split by PBC. 
        for _node in _subgraph.nodes:
            
            # first check starting point of outgoing edges for this node
            _outgoing = np.where(_sedges[:,0] == _node)[0]
            _roots_out = [_subgraph.get_edge_data(*_s)['seg'][0] for _s in _sedges[_outgoing]]
            
             # then check ending point of ingoing edges for this node
            _ingoing = np.where(_sedges[:,1] == _node)[0]
            _roots_in = [_subgraph.get_edge_data(*_s)['seg'][-1] for _s in _sedges[_ingoing]]
           
            # if the edges attached to the node have different roots, it is a dangling node
            _roots  = np.array(_roots_out + _roots_in)
            _uroots = np.unique(np.round(_roots, 6), axis=0) # it is annoying but we need to round since checking unique of floats
            _rootdist = np.linalg.norm(_uroots-_roots[0], axis=1)

            # if any supposedly connected nodes are not on top of one another, they are displaced by a periodic cell vector 
            _bool = (_rootdist > 1e-3)
            if np.sum(_bool) > 0:

                if _sid not in extended_dict: 
                    extended_dict[_sid] = np.array([False, False, False])

                # determine the box dimension along which the network is extended
                _dvecs = _uroots[_bool] - _roots[0]
                for _dvec in _dvecs:
                    extended_dict[_sid][np.abs(_dvec) > 1e-6] = True

                print ("subgraph %d is extended along dimensions x, y, z:" % _sid, extended_dict[_sid])

                # for this subgraph, wrap the xyz coordinates of each segment back into the box along the extended dimension 
                for edge in _subgraph.edges(data=True):
                    _wrapped = np.array([rfile.pbcwrap(_x) for _x in edge[2]['seg']])
                    for _bi,_extended in enumerate(extended_dict[_sid]):
                        if _extended:
                            edge[2]['seg'][:,_bi] = _wrapped[:,_bi]
    #'''

    if extended_dict != {}:
        extended = True
    else:
        extended = False
        if large_network_dict != {}:
            print ("No extended dislocations found.\n")

    # relaxation volume of complete dislocation network
    alattice = 3.1652
    omega0 = .5*alattice**3
    omegatot, omegalist = relaxationvolume(sshifted, alattice, omega0, offset=[0,0,0])

    # compute relaxation volume of translated network
    omegatot2,_ = relaxationvolume(sshifted, alattice, omega0, offset=[1e5,-1e4,-2e4], verbose=False)
    if np.abs(omegatot-omegatot2) > 10.0:
        print ("Warning: relaxation volumes with %f and without offset %f differ by more than 10 atomic volumes!\n" % (omegatot, omegatot2)) 
    else:
        print ("Relaxation volume passed check for translational invariance.\n")

    # compute relaxation volume tensor 
    volumetensor,_ = relaxationvolumetensor(sshifted, alattice, omega0, offset=[0,0,0], verbose=False)
    print ('Relaxation volume tensor in atomic volumes: ') 
    print ("%12.4f %12.4f %12.4f" % tuple(volumetensor[0]))
    print ("%12.4f %12.4f %12.4f" % tuple(volumetensor[1]))
    print ("%12.4f %12.4f %12.4f" % tuple(volumetensor[2]))
    print ()


    # walk through the segments of extended graphs and determine where they are disconnected
    boundary_points = {_k:[] for _k in range(3)} 
    boundary_bvecs  = {_k:[] for _k in range(3)} 

    for _sid in extended_dict:
        _subgraph = sshifted[_sid]
        for edge in _subgraph.edges(data=True):
            _r = edge[2]['seg']
            _dr = _r[1:] - _r[:-1]
            fdel = np.array([rfile.cmati@_dri for _dri in _dr])

            _ixb = np.where(np.abs(fdel[:,0]) >= .5)[0]
            _iyb = np.where(np.abs(fdel[:,1]) >= .5)[0]
            _izb = np.where(np.abs(fdel[:,2]) >= .5)[0]

            # plane normals
            _cx,_cy,_cz = rfile.cmat.T
            _nx = np.cross(_cy, _cz)
            _ny = np.cross(_cz, _cx)
            _nz = np.cross(_cx, _cy)
 
            _nx = _nx/np.sqrt(_nx@_nx)
            _ny = _ny/np.sqrt(_ny@_ny)
            _nz = _nz/np.sqrt(_nz@_nz)

            insert_pts = []
            insert_idx = []
            for _ik,_ikb in enumerate([_ixb, _iyb, _izb]):
                if _ikb.size > 0:
                    for _kb in _ikb:
                        _r1, _r2 = _r[_kb], _r[_kb+1]

                        # segments can be in or outgoing
                        sign = 1
                        if _r1[_ik] < _r2[_ik]:
                            _r1, _r2 = _r2, _r1
                            sign = -1

                        # restore unbroken point coordinates
                        _dr = _r2 - _r1
                        _r2 = _r1 + rfile.minimg(_dr) 

                        # determine intersection of line to plane: point r is on a plane with normal n if (r-r0).n = 0, where r0 is a pt on the plane
                        _ci = rfile.r0 + [_cx,_cy,_cz][_ik]
                        _ni = [_nx,_ny,_nz][_ik]
                        alpha = ((_r1-_ci)@_ni)/((_r1-_r2)@_ni)
                        _ris1 = _r1 + alpha*(_r2-_r1)
                        _ris2 = _ris1 - [_cx,_cy,_cz][_ik]

                        #print (_kb, _r1, _r2, _ris1, _ris2)

                        # check that resulting intersection point _ris indeed lies on the plane
                        _pcond1 = (_ris1 - _ci)@_ni
                        _pcond2 = (_ris2 - rfile.r0)@_ni
                        assert np.abs(_pcond1) < 1e-9, "Error: intersection point is not on the plane! Condition: %f" % _pcond1 
                        assert np.abs(_pcond2) < 1e-9, "Error: intersection point is not on the plane! Condition: %f" % _pcond2

                        # for closure, add the boundary point lying on planes adjacent to the (111) simulation cell corner 
                        boundary_points[_ik] += [_ris1]
                        boundary_bvecs[_ik]  += [sign*edge[2]['burgers']]
                        
                        if sign == -1:
                            _ris1, _ris2 = _ris2, _ris1

                        # add the new boundary points and their indices of insertion
                        insert_pts += [_ris1, _ris2]
                        insert_idx += [_kb+1, _kb+1]

            # actually insert the points into the segment
            if insert_pts != []:
                _r = np.insert(_r, insert_idx, insert_pts, axis=0)
                edge[2]['seg'] = _r
                #print (insert_idx, insert_pts)
                #for _row in edge[2]['seg']:
                #    print (_row)
                #print ()

    if extended:
        for _k in boundary_points.keys():
            boundary_points[_k] = np.array(boundary_points[_k])
            boundary_bvecs[_k]  = np.array(boundary_bvecs[_k])
            # consistency check
            assert (np.abs(np.sum(boundary_bvecs[_k], axis=0)) < 1e-6).all(), "Error: net Burgers vector of dislocations threading along dim %d is non-zero!" % _k
            assert len(boundary_points[_k]) != 1, "Error: found just a single threaded dislocation along dim %d! Closure impossible." % _k 

        print (boundary_points)
        print (boundary_bvecs) 

    if 0==1:
        # compute dislocation line lengths
        perimeterall, perimetertot, perimeterdict = linelength(sgraph, verbose=1)
        # save length output
        fending = fpath.split('.')[-1]
        fname = fpath.rstrip('.'+fending)
        
        exportlines = []
        for _key in perimeterdict:
            for _line in perimeterdict[_key]:
                exportlines += [[_key, _line]]
        exportlines = np.insert(exportlines, 0, np.r_[0.0, perimeterall], axis=0)
        np.savetxt('%s.line' % fname, exportlines, fmt="%10.4f %16.8f")

        return 0
 
    # compute periodic correction relaxation volume tensor 
    if extended:
        # loop over all boundary points and compute relaxation volume correction 
        print ("Computing periodic closure correction to the relaxation volume.")

        _omegaij = np.zeros((3,3))
        for _k in boundary_points.keys():
            bdpts = boundary_points[_k]  
            bvecs = boundary_bvecs[_k]
            for _i in range(1, len(bdpts)):
                _outer = -.5*np.outer(bvecs[_i], np.cross(rfile.cmat[:,_k], bdpts[_i]-bdpts[0]))
                _omegaij += .5*(_outer + _outer.T) 

        _omega = np.trace(_omegaij)
        pbcvolume, pbcvolumetensor = alattice*_omega/omega0, alattice*_omegaij/omega0 
        print ("Relaxation volume correction: %14.8f atomic volumes\n" % pbcvolume)
        print ('Periodic relaxation volume tensor correction in atomic volumes: ') 
        print ("%12.4f %12.4f %12.4f" % tuple(pbcvolumetensor[0]))
        print ("%12.4f %12.4f %12.4f" % tuple(pbcvolumetensor[1]))
        print ("%12.4f %12.4f %12.4f" % tuple(pbcvolumetensor[2]))
        print ()

        print ("PBC-corrected relaxation volume: %14.8f atomic volumes\n" % (omegatot+pbcvolume))

        print ('PBC-corrected relaxation volume tensor in atomic volumes: ') 
        print ("%12.4f %12.4f %12.4f" % tuple(volumetensor[0]+pbcvolumetensor[0]))
        print ("%12.4f %12.4f %12.4f" % tuple(volumetensor[1]+pbcvolumetensor[1]))
        print ("%12.4f %12.4f %12.4f" % tuple(volumetensor[2]+pbcvolumetensor[2]))
        print ()


    # determine real loop linesense 
    # import the complete atomic dump file
    if dpath:
        print ("\nOptional phase: validate linesense of dislocation loops using dump file.")
        dfile = ReadDump(dpath)
        dfile.load()

        # determine loop character for every dislocation
        graphmeanbonds = probe_all(sshifted, alattice, dfile, dfile.ktree, probedistance=5.0) 
        loopcharacters = determine_character(graphmeanbonds)
    else:
        loopcharacters = np.ones(len(omegalist))
        debug_xyz = None
        ndebug = 0

    # save relaxation volume output
    fending = fpath.split('.')[-1]
    fname = fpath.rstrip('.'+fending)

    exportvolumes = np.c_[np.arange(len(omegalist), dtype=int), omegalist, loopcharacters]
    if extended is not False:
        exportvolumes = np.insert(exportvolumes, 0, np.r_[-1, pbcvolume, 1], axis=0) 
    else: 
        exportvolumes = np.insert(exportvolumes, 0, np.r_[-1, 0, 1], axis=0) 

    np.savetxt('%s.dat' % fname, exportvolumes, fmt="%5d %18.6f %3d")


    if dpath:
        # export DXA dump file
        print ("\nFetching atoms near DXA points for debugging.")
        nebs = []
        for _i,_subgraph in enumerate(sshifted):
            nodes = list(_subgraph.edges)
            for _j,edge in enumerate(nodes):
                _edgedat = _subgraph.get_edge_data(edge[0], edge[1])[edge[2]]
                for _k,row in enumerate(_edgedat['seg']):
                    nebs += dfile.ktree.query_ball_point(row, 7.0)
        nebs = np.unique(nebs)
        debug_xyz = dfile.xyz[nebs] 
        ndebug = len(debug_xyz)
        print ("Exporting %d out of %d atoms." % (ndebug, dfile.natoms))

    # count number of dislocation points
    npts = 0
    for _subgraph in sshifted:
        nodes = list(_subgraph.edges)
        for edge in nodes:
            for row in _subgraph.get_edge_data(edge[0], edge[1])[edge[2]]['seg']:
                npts += 1

    efile = "%s.unfolded" % fname
    print ("Exporting processed dxa data to %s" % efile)
    wfile = open(efile, 'w')
    wfile.write("ITEM: TIMESTEP\n")
    wfile.write("0\n")
    wfile.write("ITEM: NUMBER OF ATOMS\n")
    wfile.write("%d\n" % (npts+ndebug))

    # assume 3D pbc
    if len(rfile.celldim[0]) == 2:
        wfile.write("ITEM: BOX BOUNDS pp pp pp\n")
    else:
        wfile.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
    wfile.write("%f "*len(rfile.celldim[0]) % tuple(rfile.celldim[0]) + "\n")
    wfile.write("%f "*len(rfile.celldim[1]) % tuple(rfile.celldim[1]) + "\n")
    wfile.write("%f "*len(rfile.celldim[2]) % tuple(rfile.celldim[2]) + "\n")

    wfile.write("ITEM: ATOMS id type x y z tbx tby tbz tx ty tz segmentid relaxationvolume\n")

    for _i,_subgraph in enumerate(sshifted):
        nodes = list(_subgraph.edges)
        for _j,edge in enumerate(nodes):
            _edgedat = _subgraph.get_edge_data(edge[0], edge[1])[edge[2]]
            
            # burgers vector
            _tb = _edgedat['burgers']

            # segment id 
            _segid = _edgedat['segid']

            # loop character and relaxation volume
            _omega = omegalist[_i]
            if loopcharacters[_i] < 0: 
                _omega = -_omega
            _type = np.sign(_omega)            

            # tangent vector
            _dr =  _edgedat['seg'][1:] -  _edgedat['seg'][:-1]
            _dr = np.concatenate([_dr, [np.r_[0,0,0]]])
            for _k,row in enumerate(_edgedat['seg']):
                _r = row# + rfile.r0
                wfile.write('%3d %3d %14.8f %14.8f %14.8f %6.3f %6.3f %6.3f %14.8f %14.8f %14.8f %6d %14.8f\n' % (_i, _type, 
                                                                                                                  _r[0], _r[1], _r[2], 
                                                                                                                  _tb[0], _tb[1], _tb[2], 
                                                                                                                  _dr[_k,0], _dr[_k,1], _dr[_k,2], 
                                                                                                                  _segid, _omega))
    if dpath:
        # export neighbouring atoms
        for _i,_xyz in enumerate(debug_xyz):
            _r = _xyz + rfile.cell0
            wfile.write('%3d 2 %14.8f %14.8f %14.8f 0 0 0 0 0 0 0 0\n' % (_i, _r[0], _r[1], _r[2])) 

    wfile.close()

    return 0


if __name__=="__main__":
    main ()
