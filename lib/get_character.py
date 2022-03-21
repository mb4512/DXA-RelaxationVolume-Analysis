import numpy as np
from scipy import spatial

from lib.graphstuff import pbcdistvec

def probe_bond(r, d, b, dfile, ktree, maxbond=2.9, probedistance=7.0, proberadius=5.0):
    '''Return the mean bond lengths along b-vector in a region near the core. 

    Find all atoms within proberadius at r+probedistance*d, then construct bonds of length
    at most maxbond between them. Only bonds are kept that are within 35 degrees of the
    burgers vector b. The mean bond length is returned. The probed region should be just above
    or below the slip plane, such that the mean bonds are highly distorted. 
    
    Arguments
    ---------
    r: 1x3 vector, probe origin
    d: 1x3 vector, probe direction
    b: 1x3 vector, burgers vector
    dfile: class instance from ReadFile 
    ktree: ktree instance for all atoms in dfile

    Parameters
    ----------
    maxbond: maximum bond length to accept, should be within 10 to 20 percent to b (default: 2.9)
    probedistance: distance from r at which atomic bonds are probed (default: 7.0)
    proberadius: radius of sphere within which atoms are probed (default: 5.0)
    
    Returns
    -------
    result: mean bond length in the probed region
    ''' 
    
    meanbond = [0,0]
    for si,sign in enumerate([1, -1]):
        # fetch some atoms from a region we believe to be all on one side of the slip plane
        probeids = ktree.query_ball_point(r+sign*probedistance*d, proberadius)

        # construct a ktree out of those atoms
        _pxyz = dfile.xyz[probeids]
        _pktree = spatial.cKDTree(_pxyz, boxsize=dfile.cell)

        # find bond vectors connecting atoms closer than maxbond Angstrom (1st neb.) 
        _pbondindices = _pktree.query_pairs(maxbond, output_type="ndarray")
        _pbondvectors = _pxyz[_pbondindices[:,1]] - _pxyz[_pbondindices[:,0]]

        # wrap bond vectors vectors back into the box
        _pbondvectors = pbcdistvec(_pbondvectors, dfile)

        # discard bond vectors that deviate from b-vec by more than ~35 degrees  
        _bnorm = np.linalg.norm(b)
        _pbnorm = np.linalg.norm(_pbondvectors, axis=1)
        pdotb = np.r_[[np.dot(_pb,b) for _pb in _pbondvectors]]/(_bnorm*_pbnorm)
        _bondlengths = np.linalg.norm(_pbondvectors[np.abs(pdotb) > .8], axis=1)

        # compute mean bond length
        meanbond[si] = np.mean(_bondlengths)
        
    return meanbond


def probe_all(graph, a0, dfile, ktree, probedistance=5.0):
    '''Compute mean bond for every dislocation segment for every closed loop.

    Compute (d = b cross dr) vector for checking loop character, then run probe_bond() 
    - if d points into region with shortened bonds -> compressive strain -> interstitial character
    - if d points into region with elongated bonds ->     tensile strain ->      vacancy character

    Arguments
    ---------
    graph: graph instance representing the dislocation network 
    a0: float, crystal lattice constant for maximum probe radius and bond length via 1.15*a0*norm(b)
    dfile: class instance from ReadFile 
    ktree: ktree instance for all atoms in dfile

    Parameters
    ----------
    probedistance: distance from r at which atomic bonds are probed (default: 7.0)
    
    Returns
    -------
    result: mean bond length for every dislocation segment in every loop. 
    '''

    graphmeanbonds = []

    for _i,_subgraph in enumerate(graph):
        
        graphmeanbonds += [[]]
        for edge in _subgraph.edges(data=True):
            _b = edge[2]['burgers']
            _r = edge[2]['seg']
            _dr = _r[1:]-_r[:-1]

            # filter out duplicate pts
            _dubs = np.r_[[np.any(np.abs(_dri))>1e-3 for _dri in _dr]]
            _r  = _r[1:][_dubs]
            _dr = _dr[_dubs] 
            
            _d = [np.cross(_b, _dri) for _dri in _dr]
            _d = [_di/np.dot(_di,_di) for _di in _d]

            _prad = 1.15*np.linalg.norm(_b)*a0
            _meanbonds = np.r_[[probe_bond(_r[_j], _d[_j], _b, dfile, 
                                           ktree, probedistance=probedistance, proberadius=_prad, maxbond=_prad) 
                         for _j in range(len(_r)-1)]]
            
            bp = _meanbonds.T[0][~np.isnan(_meanbonds.T[0])]
            bm = _meanbonds.T[1][~np.isnan(_meanbonds.T[1])]

            # store mean of sampled bond lengths on either side of slip plane for each dislocation segment 
            graphmeanbonds[_i] += [[np.mean(bp), np.mean(bm)]]
        graphmeanbonds[_i] = np.array(graphmeanbonds[_i])

    return graphmeanbonds


def determine_character(graphmeanbonds):
    '''Compute the largest difference between mean bonds of all segments and return loop character.

    Arguments
    ---------
    meanbonds: MxN array of mean bonds, M: number of subgraphs, N: number of segments in M

    Returns
    -------
    loopcharacter: if 1, it is consistent with linesense. If -1, it is flipped around. 
    ''' 

    print ("Mean bond lengths on either side of the slip plane for each dislocation segment.")
    loopcharacters = []
    for _i in range(len(graphmeanbonds)):
        print ("\tGraph %d:" % _i)
        for _bonds in graphmeanbonds[_i]:
            print ("\t\t%12.6f %12.6f" % tuple(_bonds))

        # if the majority of edges have compressed bonds above the slip plane, then
        # the loop character was identified correctly. Otherwise it was flipped.
        _bonddiff = graphmeanbonds[_i][:,1]-graphmeanbonds[_i][:,0]
        _right = np.sum(_bonddiff>0)
        _false = np.sum(_bonddiff<0)
        if _right > _false:
            loopcharacters += [1]
            print ("\t\t-> Consistent loop character.   %d out of %d edges have consistent line sense." % (_right, len(graphmeanbonds[_i])))
        elif _right == _false:
            loopcharacters += [1]
            print ("\t\t-> Inconclusive loop character. Half the edges have inconsistent line sense. Keeping character unchanged.")
        else:
            loopcharacters += [-1]
            print ("\t\t-> Inconsistent loop character. %d out of %d edges have inconsistent line sense." % (_false, len(graphmeanbonds[_i])))
        print ()
    loopcharacters = np.r_[loopcharacters]
    print ()

    return loopcharacters

    '''
    # fetch the largest difference between mean bonds of all segments
    bdiffpos = np.max(meanbonds[:,1]-meanbonds[:,0])
    bdiffneg = np.max(meanbonds[:,0]-meanbonds[:,1])

    if bdiffpos > bdiffneg:
        loopcharacter = 1   # loop character is consistent
    else:
        loopcharacter = -1  # loop character is flipped

    return [bdiffpos, bdiffneg, loopcharacter]
    '''
