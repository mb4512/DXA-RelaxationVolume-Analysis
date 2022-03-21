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
            _cell = np.array([[xlo,xhi], [ylo,yhi], [zlo,zhi]], dtype=np.float)
            
            _dfile.readline()

            # read in atomic coordinates
            _rawdata = [_dfile.readline().rstrip("\n").split() for i in range(natoms)]
            _rawdata = np.array(_rawdata, dtype=np.float)
            
            self.cell = _cell
            self.id = np.array(_rawdata[:,0], dtype=np.int)
        
            if dxa:
                self.xyz      = np.array(_rawdata[:,1:4], dtype=np.float)
                self.btrue    = np.array(_rawdata[:,4:7], dtype=np.float)
                self.bspatial = np.array(_rawdata[:,7:10], dtype=np.float)
                self.isloop   = np.array(_rawdata[:,-1], dtype=np.int)
            else:
                self.xyz      = np.array(_rawdata[:,2:5], dtype=np.float)
        return 0


class ReadDump:
    def __init__(self, fpath):
        self.fpath = fpath

    def load(self):
        fname = self.fpath
        t0 = time.time()

        fread = open(fname, 'r')
        linecount = 0
        for line in fread: 
            linecount = linecount+1 
        fread.close()
        print ("File %s has %d rows." % (fname, linecount))

        # pre-allocate array and load data into array
        dumpdata = np.zeros((linecount-9, 5), dtype=np.float64)

        with open(fname) as fread:
            # read in number of atoms
            fread.readline()
            fread.readline()
            fread.readline()
            natoms = int(fread.readline().rstrip())

            # read in box dimensions (assuming cubic dimensions)
            fread.readline()
            xlo,xhi = fread.readline().split()
            ylo,yhi = fread.readline().split()
            zlo,zhi = fread.readline().split()
            _cell = np.array([[xlo,xhi], [ylo,yhi], [zlo,zhi]], dtype=np.float)
            fread.readline()
           
            print ('Importing file %s with %d atoms...' % (fname, natoms))
            for i,line in tqdm(enumerate(fread), desc="importing", total=linecount-9, ncols=150):
                dumpdata[i] = np.fromstring(line, sep=" ")

        if i+1 != natoms:
            print ('Error: found %d atoms instead of %d.' % (i+1, natoms))
            return 1
        else:
            print ('Imported %d atoms after %6.3f seconds.' % (natoms, time.time()-t0))

        print ("\nSimulation cell vectors:")
        self.cell = _cell
        print (self.cell)

        self.xyz = dumpdata[:,2:] - self.cell[:,0]
        self.cell = self.cell[:,1] - self.cell[:,0]
        self.natoms = len(self.xyz)

        print ("Peek at 5 first rows:")      
        print (self.xyz[:np.min([5,natoms])])
 
        # wrap atoms back into the box and build kdtree for quick neighbour search
        print ("\nConstructing KD-tree.")
        t0 = time.time()
        self.xyz = np.r_[[_x - self.cell*np.floor(_x/self.cell) for _x in self.xyz]]
        self.ktree = spatial.cKDTree(self.xyz, boxsize=self.cell)
        print ('Completed after %6.3f seconds.' % (time.time()-t0))

        return 0
