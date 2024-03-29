# Boilerplate code generated by OVITO 3.1.3
import os, sys, glob
import numpy as np

from ovito.io import *
from ovito.modifiers import *
from ovito.pipeline import *

def main():
    # Data import:
    importstring = sys.argv[1]

    ifiles = glob.glob(importstring)
    if len(ifiles) == 0:
        print ("No files %s found in this directory." % importstring)

    if len(ifiles) == 1:
        dxafile = "dxa_%s" % importstring
        if os.path.exists(dxafile):
            print ("Skipping file %s as %s already present." % (importstring, dxafile))
            return 0 

    ### then read in all other files ###
    pipeline = import_file(importstring)#, atom_style='atomic')

    # Dislocation analysis (DXA):
    pipeline.modifiers.append(DislocationAnalysisModifier(
        trial_circuit_length = 14, 
        #circuit_stretchability = 14,
        circuit_stretchability = 200,
        input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC,
        line_point_separation = 1.0, 
        defect_mesh_smoothing_level = 0))

    logfile = open(sys.argv[2], 'w')
    logfile.write('# file frame L<111> L<100> L<other> (in Angstrom)\n')

    for frame in range(pipeline.source.num_frames):

        data = pipeline.compute(frame)
        dsegs = data.dislocations.segments
        npts = np.sum([len(seg.points) for seg in dsegs])

        sfile = data.attributes['SourceFile']
        sfile = sfile.split('/')[-1]

        print ('DXA analysis: file, #segs, #pts: %40s %8d %8d' % (sfile, len(dsegs), npts))

        # write file header
        wfile = open("dxa_%s" % sfile, 'w')
        wfile.write("ITEM: TIMESTEP\n")
        wfile.write("%d\n" % frame)
        wfile.write("ITEM: NUMBER OF ATOMS\n")
        wfile.write("%d\n" % npts)

        pbc = ""
        for _p in data.cell.pbc:
            if _p == True:
                pbc += "pp "
            else:
                pbc += "f "

        # simulation box dimensions specific to orthogonal box
        wfile.write("ITEM: BOX BOUNDS xy xz yz %s\n" % pbc)
        cell = data.cell.matrix

        Av = cell[:,0]
        Bv = cell[:,1]
        Cv = cell[:,2]
        r0 = cell[:,3]
        A,B,C = np.linalg.norm(Av),  np.linalg.norm(Bv),  np.linalg.norm(Cv)

        ax = A 
        bx = np.dot(Bv, Av/A) 
        by = np.sqrt(B*B - bx*bx)
        cx = np.dot(Cv, Av/A)
        cy = (np.dot(Bv, Cv) - bx*cx)/by
        cz = np.sqrt(C*C - cx*cx - cy*cy)

        xlo,ylo,zlo = r0 
        xhi,yhi,zhi = xlo+ax, ylo+by, zlo+cz
        xy,xz,yz = bx,cx,cy

        xlo_bound = xlo + min(0.0,xy,xz,xy+xz)
        xhi_bound = xhi + max(0.0,xy,xz,xy+xz)
        ylo_bound = ylo + min(0.0,yz)
        yhi_bound = yhi + max(0.0,yz)
        zlo_bound = zlo
        zhi_bound = zhi
         
        wfile.write("%f %f %f\n" % (xlo_bound, xhi_bound, xy)) 
        wfile.write("%f %f %f\n" % (ylo_bound, yhi_bound, xz)) 
        wfile.write("%f %f %f\n" % (zlo_bound, zhi_bound, yz)) 

        wfile.write("ITEM: ATOMS id x y z tbx tby tbz sbx sby sbz loop\n")

        nd = len(dsegs)
        if len(dsegs) > 0:
            # first, write down dislocation line coordinates 
            for si,seg in enumerate(dsegs):
                for pt in seg.points:
                    btrue    = seg.true_burgers_vector
                    bspatial = seg.spatial_burgers_vector
                    #print (si, pt[0], pt[1], pt[2], btrue[0], btrue[1], btrue[2], bspatial[0], bspatial[1], bspatial[2], seg.is_loop)

                    wfile.write("%5d %12.7f %12.7f %12.7f %8.4f %8.4f %8.4f %12.7f %12.7f %12.7f %3d\n" % (si, 
                                                             pt[0], pt[1], pt[2], 
                                                             btrue[0], btrue[1], btrue[2], 
                                                             bspatial[0], bspatial[1], bspatial[2],
                                                             int(seg.is_loop)))
            # then, write down dislocation line stats
            dlengths = np.r_[[d.length for d in dsegs]]
            dburgers = [d.true_burgers_vector for d in dsegs]
            dbnorm   = np.linalg.norm(dburgers, axis=1)

            b111 = np.sqrt(3)/2.
            b001 = 1.0 
            ix111 = np.where(np.abs(dbnorm-b111) < .1)[0] 
            ix001 = np.where(np.abs(dbnorm-b001) < .1)[0] 

            len111 = np.sum(dlengths[ix111])
            len001 = np.sum(dlengths[ix001])
            lenetc = np.sum(dlengths) - len111 - len001
        else:
            len111 = 0.0 
            len001 = 0.0 
            lenetc = 0.0 

        write_string = "%15.7f %15.7f %15.7f" % (len111, len001, lenetc)
        logfile.write('%30s %8d %s\n' % (sfile, 1+frame, write_string))


        wfile.close()
    logfile.close()



if __name__=="__main__":
    main()
