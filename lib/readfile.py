import numpy as np
import time
from scipy import spatial

from tqdm import tqdm

class ReadFile:
    def __init__(self, fpath):
        self.fpath = fpath
        
    def load(self, dxa=True):
        self.read_dump(self.fpath, dxa)

        self.xyz    = self.xyz - self.cell[:,0]
        self.cell0  = self.cell[:,0] 
        self.cell   = self.cell[:,1] - self.cell[:,0]
        self.natoms = len(self.xyz)

        if dxa:
            # split into segments
            maxid = np.max(self.id)
            segments = [np.where(_id == self.id)[0] for _id in range(maxid+1)]
            self.xyz = [self.xyz[_seg] for _seg in segments]

            self.btrue    = np.r_[[self.btrue[_seg][0]    for _seg in segments]]
            self.bspatial = np.r_[[self.bspatial[_seg][0] for _seg in segments]]
            self.isloop   = np.r_[[self.isloop[_seg][0]   for _seg in segments]]
            self.id = np.unique(self.id)

            self.nsegments = len(segments)
            print ("Imported %d points and %d segments from file: %s" % (self.natoms, self.nsegments, self.fpath))
        else:
            print ("Imported %d atoms from file: %s" % (self.natoms, self.fpath))
            
        print ()

    def read_dump(self, fpath, dxa):
        with open(fpath, 'r') as _dfile:
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            natoms = int(_dfile.readline())

            _dfile.readline()

            # read in box dimensions (assuming cubic dimensions)
            xlo,xhi = _dfile.readline().split()
            ylo,yhi = _dfile.readline().split()
            zlo,zhi = _dfile.readline().split()
            _cell = np.array([[xlo,xhi], [ylo,yhi], [zlo,zhi]], dtype=float)
            
            _dfile.readline()

            # read in atomic coordinates
            _rawdata = [_dfile.readline().rstrip("\n").split() for i in range(natoms)]
            _rawdata = np.array(_rawdata, dtype=float)
            
            self.cell = _cell
            self.id = np.array(_rawdata[:,0], dtype=int)
        
            if dxa:
                self.xyz      = np.array(_rawdata[:,1:4], dtype=float)
                self.btrue    = np.array(_rawdata[:,4:7], dtype=float)
                self.bspatial = np.array(_rawdata[:,7:10], dtype=float)
                self.isloop   = np.array(_rawdata[:,-1], dtype=int)
            else:
                self.xyz      = np.array(_rawdata[:,2:5], dtype=float)
        return 0


class ReadDump:
    def __init__(self, fpath):
        self.fpath = fpath

    def load(self, dxa=False):
        fname = self.fpath
        t0 = time.time()

        fread = open(fname, 'r')
        linecount = 0
        for line in fread: 
            linecount = linecount+1 
        fread.close()
        print ("File %s has %d rows." % (fname, linecount))

        with open(fname) as fread:
            # read in number of atoms
            fread.readline()
            fread.readline()
            fread.readline()
            self.natoms = int(fread.readline().rstrip())

            # read in box dimensions from the next 3 lines
            fread.readline()
            celldim = np.array([fread.readline().split() for i in range(3)], dtype=float)
            self.celldim = celldim # save for export later

            # all systems are treated as triclinic by default
            # if imported cell is orthorhombic, add zero-valued xy, xz, yz values 
            if celldim.shape == (3,2):
                xlo_bound, xhi_bound = celldim[0] 
                ylo_bound, yhi_bound = celldim[1] 
                zlo_bound, zhi_bound = celldim[2] 
                xy,xz,yz = 0.,0.,0.
            else:
                xlo_bound, xhi_bound, xy = celldim[0]
                ylo_bound, yhi_bound, xz = celldim[1]
                zlo_bound, zhi_bound, yz = celldim[2]

            xlo = xlo_bound - min(0., xy, xz, xy+xz)
            xhi = xhi_bound - max(0., xy, xz, xy+xz)
            ylo = ylo_bound - min(0., yz)
            yhi = yhi_bound - max(0., yz)
            zlo = zlo_bound
            zhi = zhi_bound

            # construct cell origin 
            self.r0 = np.r_[xlo, ylo, zlo]

            # construct cell vectors
            self.c1 = np.r_[xhi-xlo, 0., 0.]
            self.c2 = np.r_[xy, yhi-ylo, 0.]
            self.c3 = np.r_[xz, yz, zhi-zlo]
            self.cmat = np.c_[[self.c1,self.c2,self.c3]].T

            # fetch number of columns to import
            header = fread.readline().split(' ')
            ncols = len(header)-2

            # pre-allocate array and load data into array
            _rawdata = np.zeros((linecount-9, ncols), dtype=float)
              
            print ('Importing file %s with %d particles...' % (fname, self.natoms))
            for i,line in tqdm(enumerate(fread), desc="importing", total=linecount-9, ncols=150):
                _rawdata[i] = np.fromstring(line, sep=" ")

        if i+1 != self.natoms:
            print ('Error: found %d particles instead of %d.' % (i+1, self.natoms))
            return 1
        else:
            print ('Imported %d particles after %6.3f seconds.' % (self.natoms, time.time()-t0))

        self.id = np.array(_rawdata[:,0], dtype=int)
        if dxa:
            self.xyz      = np.array(_rawdata[:,1:4], dtype=float)
            self.btrue    = np.array(_rawdata[:,4:7], dtype=float)
            self.bspatial = np.array(_rawdata[:,7:10], dtype=float)
            self.isloop   = np.array(_rawdata[:,-1], dtype=int)
        else:
            self.xyz      = np.array(_rawdata[:,2:5], dtype=float)

        print ("\nSimulation cell origin:")
        print (self.r0)

        print ("\nSimulation cell vectors:")
        print (self.cmat)

        # get inverse matrices for lattice vector transformation
        self.cmati = np.linalg.inv(self.cmat)
      
        if dxa:
            # split into segments
            maxid = np.max(self.id)
            segments = [np.where(_id == self.id)[0] for _id in range(maxid+1)]
            self.xyz = [self.xyz[_seg] for _seg in segments]

            self.btrue    = np.r_[[self.btrue[_seg][0]    for _seg in segments]]
            self.bspatial = np.r_[[self.bspatial[_seg][0] for _seg in segments]]
            self.isloop   = np.r_[[self.isloop[_seg][0]   for _seg in segments]]
            self.id = np.unique(self.id)

            self.nsegments = len(segments)
            print ("\nImported %d points and %d segments from file: %s" % (self.natoms, self.nsegments, self.fpath))
        else:
            print ("\nImported %d atoms from file: %s" % (self.natoms, self.fpath))            
        print ()

        '''
        # wrap atoms back into simulation cell if outside
        #self.xyz = np.r_[[self.pbcwrap1(_x) for _x in self.xyz]]

        # build KDTree of periodically repeated reference structure for nearest neighbour search
        print ("\nConstructing KD-tree.")
        t0 = time.time()

        # build KDTree of periodically repeated reference structure for nearest neighbour search
        self.kindices, self.kdtree = self.build_triclinic_kdtree(self.xyz)

        print ('Completed after %6.3f seconds.' % (time.time()-t0))
        '''

        return 0


    def pbcwrap(self, xyz):
        '''Check if a vector falls outside the box, and if so, wrap it back inside.'''
        fcoords = np.matmul(self.cmati, xyz - self.r0)
        gcoords = fcoords - np.floor(fcoords)
        return self.r0 + np.matmul(self.cmat, gcoords)

    def build_triclinic_kdtree(self, xyz):
        ids = np.r_[:len(xyz)]
        icopy = np.copy(ids)
        xcopy = np.copy(xyz)
        for ix in [-1,0,1]:
            for iy in [-1,0,1]:
                for iz in [-1,0,1]:
                    if ix == iy == iz == 0:
                        continue
                    _xcopy = np.copy(xyz)
                    _xcopy += ix*self.c1+ iy*self.c2 + iz*self.c3
                    xcopy = np.r_[xcopy, _xcopy]
                    icopy = np.r_[icopy, ids]

        return icopy, xcopy, spatial.cKDTree(xcopy, copy_data=True)


    def ktri_nebs(self, *args, **kwargs):
        res = self.ktree.count_neighbors(*args, **kwargs)
 



