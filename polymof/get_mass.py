#!/usr/bin/env python

import sys
import argparse
from pathlib import Path
from flip_hydrogens import readPDB

def parse_args():
    parser = argparse.ArgumentParser(
            description="Compute mass and density for a PDB file"
    )
    parser.add_argument("pdb_file", help="PDB file")
    args = parser.parse_args()

    return parser.parse_args()

def main():
    args = parse_args()
    rv = get_mass_for_pdb(args.pdb_file)
    print("mass:", rv['totalMass'])
    if 'L' in rv:
        print("Box length:", rv['L'])
        print("volume:", rv['V'])
        print("density:", rv['density'])
        print("density(g/cc):", rv['density']*1e24/6.02214076e23)

def get_mass_for_pdb(pdb_file):
    if not Path(pdb_file).is_file():
        print("PDB file", pdb_file, "must exist")
    data = readPDB(pdb_file)
    masses = {"H": 1.0079, "C": 12.011, "N": 14.007, "Fe": 55.845, "Zn": 65.38}
    totalMass = 0
    for e in data.elements.values():
        if e not in masses:
            print("Unknown mass for", e)
            sys.exit(1)
        totalMass += masses[e]
    cryst = [l for l in data.preATOM if "CRYST1" in l]
    if len(cryst) == 1:
        bits = cryst[0].split()
        L = [float(bits[i]) for i in range(1,4)]
        V = L[0]*L[1]*L[2]
        density = totalMass / V
        return {"totalMass": totalMass, "L": L, "V": V, "density": density}
    else:
        return {"totalMass": totalMass}

if __name__ == "__main__":
    main()
