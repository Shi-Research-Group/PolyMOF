#!/usr/bin/env python

import sys
import argparse
from pathlib import Path
from flip_hydrogens import readPDB, writePDB
import numpy as np
import math

def parse_args():
    parser = argparse.ArgumentParser(
            description="Take PBI oligomer PDB as input and writes out a new PDB with a longer oligmer by adding more monomers to one end of the oligmer."
    )
    parser.add_argument("input_pdb", help="PDB file containing input PBI oligomer")
    parser.add_argument("output_pdb", help="PDB file to write with flipped hydrogens")
    parser.add_argument("--bond", required=True, help="Indicates the bond (A:B) to break.  Atom A is in the monomer while atom B is not.")
    parser.add_argument("--polymerize", type=int, help="Number of repeat units in new oligomer")
    parser.add_argument("--attach", type=int, help="Atom C, where atom A will be attahced when making oligomer.  Additional monomers will be translated so that the new A-C bond vector is in the same direction as the old A-B bond")
    args = parser.parse_args()
    bond = args.bond.split(":")
    if len(bond) != 2:
        parser.error("Bond syntax is A:B where A and B are atom numbers.  A is in the monomer and B is not")
    args.bond = [int(bond[i]) for i in range(2)]

    return args

def walkFrom(dataPDB, atomStart, atomsExclude):
    fragmentAtoms = []
    atomsSeen = atomsExclude
    stack = [atomStart]
    while len(stack) > 0:
        a = stack.pop()
        for b in dataPDB.bonds[a]:
            if b in atomsSeen:
                continue
            fragmentAtoms.append(b)
            atomsSeen.append(b)
            stack.append(b)
    return fragmentAtoms

def filterAtoms(dataPDB, atoms):
    dataPDB.atomLines = [l for l in dataPDB.atomLines if int(l.split()[1]) in atoms]
    dataPDB.bonds = {a: [b for b in dataPDB.bonds[a] if b in atoms] for a in atoms}
    dataPDB.elements = {a: dataPDB.elements[a] for a in atoms}
    dataPDB.preCONECT[0] = f"TER   {len(atoms):5d}\n"
    return dataPDB

def renumberData(dataPDB, numberMap):
    al = dataPDB.atomLines
    dataPDB.atomLines = [al[i][:6]+f"{i+1:5d}"+al[i][11:] for i in range(len(al))]
    b = dataPDB.bonds
    dataPDB.bonds = {numberMap[i]:[numberMap[b[i][j]] for j in range(len(b[i]))] for i in b}
    c = dataPDB.coords
    dataPDB.coords = {numberMap[i]:c[i] for i in c if i in numberMap}
    e = dataPDB.elements
    dataPDB.elements = {numberMap[i]:e[i] for i in e if i in numberMap}

def polymerize(dataPDB, atomA, atomC, xyzShift, n):
    fragSize = len(dataPDB.atomLines)
    al = dataPDB.atomLines
    b = dataPDB.bonds
    c = dataPDB.coords
    e = dataPDB.elements
    finalBonds = []
    for i in range(1,n):
        ishift = i*fragSize
        for j in range(1,fragSize+1):
            xyz = [c[j][0]+xyzShift[0]*i, c[j][1]+xyzShift[1]*i, c[j][2]+xyzShift[2]*i]
            c[j+ishift] = xyz
            l = al[j-1]
            al.append("{}{:5d}{}{:8.3f}{:8.3f}{:8.3f}{}".format(l[:6], j+ishift, l[11:30], xyz[0], xyz[1], xyz[2], l[54:]))
            nb = [b[j][k]+ishift for k in range(len(b[j]))]
            if j == atomA:
                bbb = atomC+(i-1)*fragSize
                nb.append(bbb)
                finalBonds.append([bbb,j+ishift])
            b[j+ishift] = nb
            e[j+ishift] = e[j]
    for i,j in finalBonds:
        b[i].append(j)
    dataPDB.preCONECT[0] = f"TER   {len(dataPDB.atomLines):5d}\n"

def addToAtoms(dataPDB, atomLine, coords, bonds, element):
    i = len(dataPDB.atomLines) + 1
    dataPDB.coords[i] = coords
    dataPDB.bonds[i] = bonds
    for j in bonds:
        dataPDB.bonds[j].append(i)
    dataPDB.elements[i] = element
    l = atomLine
    dataPDB.atomLines.append("{}{:5d}{}{:8.3f}{:8.3f}{:8.3f}{}".format(l[:6], i, l[11:30], coords[0], coords[1], coords[2], l[54:]))

def main():
    args = parse_args()
    if not Path(args.input_pdb).is_file():
        print("input_pdb must exist")
    data = readPDB(args.input_pdb)
    xyzShift = None
    exclude = [args.bond[1]]
    if args.attach:
        c = data.coords
        xyzShift = [c[args.attach][j] - c[args.bond[1]][j] for j in range(3)]
        excludeH = [i for i in data.bonds[args.attach] if data.elements[i] == 'H']
        if len(excludeH) != 1:
            sys.exit("Could not find Hydrogen bonded to "+str(args.attach))
        excludeH = excludeH[0]
        exclude.append(excludeH)
        hVec1 = [c[excludeH][i] - c[args.attach][i]  for i in range(3)]
        hVec1Mag = math.sqrt(sum([hVec1[i]**2 for i in range(3)]))
        # we will also need this later
        hVec2 = [c[args.bond[1]][i] - c[args.bond[0]][i]  for i in range(3)]
        hVec2Mag = math.sqrt(sum([hVec2[i]**2 for i in range(3)]))
        hVec2 = [hVec2[i] * hVec1Mag / hVec2Mag for i in range(3)]
        excludeLine = data.atomLines[excludeH-1]

    monomerAtoms = walkFrom(data, args.bond[0], exclude)
    data = filterAtoms(data, monomerAtoms)
    numberMap = {int(data.atomLines[i][6:11]): i+1 for i in range(len(monomerAtoms))}
    renumberData(data, numberMap)
    if args.attach:
        polymerize(data, numberMap[args.bond[0]], numberMap[args.attach], xyzShift, args.polymerize)
        ca = data.coords[numberMap[args.attach] + (args.polymerize-1)*len(monomerAtoms)]
        hCoord1 = [ca[i] + hVec1[i] for i in range(3)]
        # we excluded a hydrogen from our monomer originally.  add it back to the end of the polymer
        addToAtoms(data, excludeLine, hCoord1, [numberMap[args.attach] + (args.polymerize-1)*len(monomerAtoms)], 'H')
        # we are also misisng a hydrogen from the other end that should be approximately where args.bond[1] was
        # add a new one with same bond length as hVec1, but pointing in the direction of args.bond[1]
        ca = data.coords[numberMap[args.bond[0]]]
        hCoord2 = [ca[i] + hVec2[i] for i in range(3)]
        addToAtoms(data, excludeLine, hCoord2, [numberMap[args.bond[0]]], 'H')
    writePDB(data, args.output_pdb)

if __name__ == "__main__":
    main()
