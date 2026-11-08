#!/usr/bin/env python

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(
            description="Take PBI oligomer PDB as input and writes out a new PDB with hydrogens put next to nitrogen pairs according to a bitmask from the commandline."
    )
    parser.add_argument("input_pdb", help="PDB file containing input PBI oligomer")
    parser.add_argument("output_pdb", help="PDB file to write with flipped hydrogens")
    parser.add_argument("hydrogens", help="Bitmask to specify hydrogen positions.  Starting from the oligomer side where the nitrogens are, 0 bit indicates H is closer to the bridge between benzenes, 1 indicates H is farther away", type=int)
    args = parser.parse_args()
    if args.hydrogens < 0:
        parser.error("hydrogens must be positive")

    return parser.parse_args()

def main():
    args = parse_args()
    if not Path(args.input_pdb).is_file():
        print("input_pdb must exist")
    data = readPDB(args.input_pdb)
    adjustHydrogens(data, int(args.hydrogens))
    writePDB(data, args.output_pdb)

@dataclass
class pdbData:
    preATOM: list
    atomLines: list
    preCONECT: list
    postCONECT: list
    elements: dict
    coords: dict
    nNums: list
    nBonds: dict
    bonds: dict

def readPDB(input_pdb):
    preATOM = []
    preCONECT = []
    postCONECT = []
    atomLines = []
    coords = {}
    nNums = []
    elements = {}
    nBonds = {}
    bonds = {}
    with open(input_pdb, "r") as f:
        foundAtoms = foundBonds = False
        for l in f:
            if len(l) > 6 and (l[:4] == "ATOM" or l[:6] == "HETATM"):
                foundAtoms = True
                atomLines.append(l)
                iAtom = int(l[6:12])
                coords[iAtom] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
                element = l[76:78].strip()
                elements[iAtom] = element
                if element == "N":
                    nNums.append(iAtom)
            elif len(l) > 6 and l[:6] == "CONECT":
                foundBonds = True
                iAtom = int(l[6:11])
                nb = (len(l)-11) // 5
                bonds[iAtom] = [int(l[11+i*5:16+i*5]) for i in range(nb)]
            elif not foundAtoms:
                preATOM.append(l)
            elif not foundBonds:
                preCONECT.append(l)
            else:
                postCONECT.append(l)
    return pdbData(preATOM, atomLines, preCONECT, postCONECT,
            elements, coords, nNums, nBonds, bonds)

def findEndNitrogen(data):
    # End nitrogens will have a C between them and the C will have an H
    # we return the N without an H
    for n in data.nNums:
        hList = [i for i in data.bonds[n] if data.elements[i] == "H"]
        if len(hList) > 0:
            # let's grab the N without an H
            continue
        cList = [i for i in data.bonds[n] if data.elements[i] == "C"]
        # our C will have an H
        for c in cList:
          chList = [i for i in data.bonds[c] if data.elements[i] == "H"]
          if len(chList) == 1:
              return n
 
    raise Exception("Could not find terminal N bonded to C bonded to H")

def findNH(data, n):
    c = [i for i in data.bonds[n] if data.elements[i] == "C"]
    n2 = [i for i in data.bonds[c[0]] if data.elements[i] == "N" and i!=n]
    if len(n2) == 0 and len(c) > 1:
      n2 = [i for i in data.bonds[c[1]] if data.elements[i] == "N" and i!=n]
    n2 = n2[0]
    h = [i for i in data.bonds[n2] if data.elements[i] == "H"][0]
    return (n2,h)

def findAllNitrogens(data, endN):
    # walk from endN
    # look for N without H, refuse to walk past N with H
    # ignore H
    # we return a list of all N without an H and list of bits
    #   0 indicates 3 bonds between the N and the nearest bridge C
    #   1 indicates 4 bonds between the N and the nearest bridge C
    pending = [[], data.bonds[endN].copy()]
    allN = [endN]
    allBits = []
    seen = {endN: True}
    while True:
        a = None
        bondsAway = -1
        for i in range(1,len(pending)):
            if len(pending[i]) > 0:
                a = pending[i].pop()
                bondsAway = i
                break
        if a == None:
            break
        if a in seen:
            continue
        seen[a] = True
        ael = data.elements[a]
        if ael == "H":
            continue
        if ael == "C":
            bridgeC = len(data.bonds[a]) == 3 and all(data.elements[bc]=="C" for bc in data.bonds[a])
            if bridgeC:
                if len(allBits) < len(allN):
                    # we found the first bridge after our previous N
                    if bondsAway < 3 or bondsAway > 4:
                        raise Exception("Found bridge C", bondsAway, "from N.  Should be 3 or 4")
                    allBits.append(bondsAway-3)
                
                pending = [[], data.bonds[a].copy()]
            elif bondsAway+1 == len(pending):
                pending.append(data.bonds[a].copy())
            else:
                pending[bondsAway+1] += data.bonds[a]
            continue
        if ael != "N":
            raise Exception("Unexpected element ", ael)
        hList = [i for i in data.bonds[a] if data.elements[i] == "H"]
        if len(hList) > 0:
            # we refuse to walk past N with H
            continue
        allN.append(a)
        if len(allN) % 2 == 0:
            if (bondsAway < 3 or bondsAway > 4) and len(allN) % 2 == 0:
                raise Exception("Found N", a, ", ", bondsAway, "bridge C.  Should be 3 or 4")
            else:
                allBits.append(bondsAway-3)
        pending = [[], data.bonds[a].copy()]
    return allN, allBits

def countHydrogens(data):
    endN = findEndNitrogen(data)
    (allN,allBits) = findAllNitrogens(data, endN)
    return len(allN)

def adjustHydrogens(data, hydrogens):
    endN = findEndNitrogen(data)
    (allN,allBits) = findAllNitrogens(data, endN)
    if hydrogens > 0:
        nbits = int(math.log2(hydrogens))
        if nbits >= len(allN):
            raise Exception("Hydrogen bits "+str(nbits)+" exceed number of nitrogen groups "+str(len(allN)))
    for i,n in enumerate(allN):
        bit = allBits[i]
        if (hydrogens>>i & 1) != bit:
            (n2,h) = findNH(data, n)
            # transfer H bond from N2 to N
            data.bonds[n].append(h)
            data.bonds[n2] = [x for x in data.bonds[n2] if x!=h]
            data.bonds[h] = [n]
            # now place H next to N
            nxyz = np.array(data.coords[n])
            n2xyz = np.array(data.coords[n2])
            hxyz = np.array(data.coords[h])
            nn2 = nxyz-n2xyz
            n2h = hxyz - n2xyz
            n2hproj = nn2 * np.dot(n2h, nn2) / np.dot(nn2, nn2)
            n2hperp = n2h - n2hproj
            hxyz = nxyz - n2hproj + n2hperp
            data.coords[h] = tuple(hxyz)

def writePDB(data, output_pdb):
    with open(output_pdb, "w") as f:
        for l in data.preATOM:
            f.write(l)
        allAtoms = []
        for l in data.atomLines:
            iAtom = int(l[6:12])
            allAtoms.append(iAtom)
            if data.elements[iAtom] != "H":
                f.write(l)
            else:
                xyz = data.coords[iAtom]
                f.write("{}{:8.3f}{:8.3f}{:8.3f}{}".format(l[:30], xyz[0], xyz[1], xyz[2], l[54:]))
        for l in data.preCONECT:
            f.write(l)
        for a in allAtoms:
            f.write("CONECT{:5d}".format(a))
            for b in data.bonds[a]:
                f.write("{:5d}".format(b))
            f.write("\n")
        for l in data.postCONECT:
            f.write(l)


if __name__ == "__main__":
    main()
