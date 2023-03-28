#!/usr/local/bin/python3
import sys, glob

from lib.readfile import *
from lib.dxa_process import *
from lib.graphstuff import *
from lib.get_character import *

def main():

    fpath = sys.argv[1]
    if len(sys.argv)>2:
        dpath = sys.argv[2]
    else:
        dpath = None

    # import dxa data file    
    rfile = ReadFile(fpath)
    rfile.load()

    # connect terminating segments
    for i in range(3):
        de11,de12,de21,de22,endsw1,endsw2,successflag = link_network(rfile, dmax=15.0)
        if successflag is True:
            break

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

    # smoothen dislocation nodes with degree 2 and create a list of all subgraphs in the network
    gsmooth = flipGraph(graph, rfile)

    if gsmooth == 1:
        print ("Failed to smoothen graph.")
        return 1
    export_graph(gsmooth, path="graph_smooth.log")
    '''

    if kirchhoff_check(graph) == 1:
        return 1

    # ensure segments are continuous across PBC
    segment_continuity(graph, rfile)

    # unwrap segments
    sgraph = [nx.MultiDiGraph(deepcopy(graph.subgraph(c))) for c in nx.weakly_connected_components(graph)]
    sshifted = relink_graph(sgraph)

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

    # relaxation volume of complete dislocation network
    alattice = 3.16
    omega0 = .5*alattice**3
    omegatot, omegalist = relaxationvolume(sshifted, alattice, omega0, offset=[0,0,0])

    # compute relaxation volume    
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
 
    # compute periodic correction relaxation volume tensor 
    pbccorrection = pbc_volume_correction(sshifted, alattice, omega0, rfile)
    if pbccorrection is not False:
        pbcvolume, pbcvolumetensor = pbccorrection

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
    if pbccorrection is not False:
        exportvolumes = np.insert(exportvolumes, 0, np.r_[-1, pbcvolume, 1], axis=0) 
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

    # assume box is orthogonal and 3D pbc
    wfile.write("ITEM: BOX BOUNDS pp pp pp\n")
    wfile.write("%f %f\n" % (rfile.cell0[0], rfile.cell[0]))
    wfile.write("%f %f\n" % (rfile.cell0[1], rfile.cell[1]))
    wfile.write("%f %f\n" % (rfile.cell0[2], rfile.cell[2]))

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
                _r = row + rfile.cell0
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
