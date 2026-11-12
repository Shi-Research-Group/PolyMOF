#!/usr/bin/env python

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
            description="Extract charges from FeIMx.mol file"
    )
    parser.add_argument("mol2_file", help="mol2 file")
    args = parser.parse_args()

    if not Path(args.mol2_file).is_file():
        parser.error("mol2 file must exist")
    charges = extract_mol2_charges(args.mol2_file)
    offset = 0
    if len(charges) == 10:
        offset = 1
    for i in range(offset,len(charges)):
        print(i-offset+1, charges[i])

def extract_mol2_charges(mol2_file):
    with open(mol2_file, "r") as f:
        charges = []
        go_atoms = False
        offset=0
        for l in f:
            if "TRIPOS>ATOM" in l:
                go_atoms = True
                continue
            if "TRIPOS>BOND" in l:
                break
            if go_atoms:
                bits = l.split()
                id=int(bits[0])
                if id == 1 and bits[5] == "Fe":
                    offset=id
                if id <= 9+offset:
                    charges.append(bits[8])
        return charges

if __name__ == "__main__":
    main()
