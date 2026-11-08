#!/usr/bin/env python

import sys
from pathlib import Path
import argparse

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

parser = argparse.ArgumentParser(
    description="Extract optimized coordinates from Gaussian log file"
)
parser.add_argument("log_file", help="Gaussian log file")
parser.add_argument("--gjf", help="Gaussian input file, coords will be updated with optimized coordaintes")
parser.add_argument("--xyz", help="XYZ file where coordinates should be written")
parser.add_argument("--allxyz", help="XYZ file where coordinates for all configurations should be written")
parser.add_argument("--fail", action='store_true', help="Print last coordinates even if Gaussian did not finish successfully")
args = parser.parse_args()

if not Path(args.log_file).is_file():
    parser.error("Log file must exist")

coords = []
atomicnumbers = []
gsuccess = False
lastline = None
optcomplete = []
with open(args.log_file, "r") as f:
    i = 0
    readcoords = -1
    for l in f:
        lastline = l
        i+=1
        if "Optimization completed" in l:
            optcomplete.append(i)
            continue
        if "Normal termination of Gaussian" in l:
            gsuccess = True
            break
        if "Standard orientation" in l:
            readcoords = 4
            icoords = []
            atomicnumbers = []
            continue
        if readcoords == 0:
            if "----" in l:
                coords.append((i,icoords))
                readcoords = -1
                continue
            bits = l.split()
            icoords.append([bits[3],bits[4],bits[5]])
            atomicnumbers.append(int(bits[1]))
            continue
        if readcoords > 0:
            readcoords-=1
            continue

if not gsuccess:
    print("Gaussian failed:\n"+lastline)
    if not args.fail:
        sys.exit(1)

if len(coords) == 0:
    print("Couldn't find any coordinates")
    sys.exit(1)

if len(optcomplete) == 0:
    print("Optimization did not seem to succeed")
    if not args.fail:
        sys.exit(1)

il,icoords = coords[len(coords)-1]

if args.gjf:
   gjflines = []
   with open(args.gjf,"r") as gin:
       i=0
       for l in gin:
           bits = l.split()
           if len(bits) == 4 and is_number(bits[1]) and is_number(bits[2]) and is_number(bits[3]):
               gjflines.append(bits[0]+" "+icoords[i][0]+" "+icoords[i][1]+" "+icoords[i][2]+"\n")
               i += 1
           else:
               gjflines.append(l)
   with open(args.gjf, "w") as gout:
       for l in gjflines:
            gout.write(l)

if args.allxyz:
   elements = {1: "H", 6: "C", 7: "N", 26: "Fe"}
   with open(args.allxyz, "w") as xyzout:
       for i in range(len(coords)):
           il,icoords = coords[i]
           xyzout.write(str(len(icoords))+"\n")
           xyzout.write("# "+args.log_file+" "+str(il)+"\n")
           for i in range(len(icoords)):
               xyzout.write(elements[atomicnumbers[i]]+" "+(" ".join(icoords[i])+"\n"))
elif args.xyz:
   elements = {1: "H", 6: "C", 7: "N", 26: "Fe"}
   with open(args.xyz, "w") as xyzout:
       xyzout.write(str(len(icoords))+"\n")
       xyzout.write("# "+args.log_file+"\n")
       for i in range(len(icoords)):
           xyzout.write(elements[atomicnumbers[i]]+" "+(" ".join(icoords[i])+"\n"))

if not args.allxyz and not args.xyz and not args.gjf:
    for xyz in icoords:
        print(xyz)
