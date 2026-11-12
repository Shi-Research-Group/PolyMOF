#!/usr/bin/env python

import sys
import flip_hydrogens
from reorder_imidazole import reorder_imidazole
import argparse
from pathlib import Path
import os
import glob
import time
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(
            description="Read the lists file, create any needed tm subdirectories, create pdb file and set up antechamber run.  Report any subdirectories that still need antechamber to run."
    )
    parser.add_argument("--lists", help="File containing a list of files with a list pentamer numbers", default="lists")
    parser.add_argument("--molecule", required=True, help="Name of molecule (pentamer).")
    parser.add_argument("--submit", action="store_true", help="Actually submit jobs as needed.")
    return parser.parse_args()

def collectH(lists):
    allH = {}
    for filename in lists:
        filename = filename.strip()
        with open(filename, "r") as f:
            for l in f:
                bits = l.split()
                if len(bits) == 1:
                    h = bits[0]
                elif len(bits) == 2:
                    print(filename, "contains 2-column h list")
                    sys.exit(1)
                    h = int(bits[1])
                else:
                    print("list file should contain 1 or 2 columns")
                    print(l)
                    sys.exit(1)
                if not h in allH:
                    allH[h] = []
                allH[h].append(filename)
    return sorted(list(allH))

def ensureAllH(allH, molecule, submit):
    now = time.time()
    allfinished = glob.glob(molecule+"*/"+molecule+"*frcmod")
    pdbdata = flip_hydrogens.readPDB(molecule+".pdb")
    numH = flip_hydrogens.countHydrogens(pdbdata)
    topdir = os.getcwd()
    prmtopdone = 0
    for h in allH:
         subd = molecule+h
         if not Path(subd).is_dir():
             os.mkdir(subd)
         hpdb = subd+"/"+molecule+h+".pdb"
         if not Path(hpdb).is_file():
             pdbdata = flip_hydrogens.readPDB(molecule+".pdb")
             flip_hydrogens.adjustHydrogens(pdbdata, int(h))
             flip_hydrogens.writePDB(pdbdata, hpdb)
             reorder_imidazole(hpdb, hpdb)
         if Path(subd+"/"+molecule+h+".prmtop").is_file():
             prmtopdone += 1
             continue
         if Path(subd+"/"+molecule+h+".frcmod").is_file():
             continue
         if not Path(subd+"/"+molecule+h+".mol2").is_file():
             if Path(subd+"/sqm.out").is_file():
                 ti = os.path.getmtime(subd+"/sqm.out")
                 if ti > now + 24*3600:
                     hours = (ti - now) // 3600
                     print("Job started "+str(hours)+" ago")
             else:
                 print("Need to submit antechamber job for "+h)
                 if not Path(subd+"/jac"+h).is_file() and Path("jac").is_file():
                     os.symlink("../jac", subd+"/jac"+h)
                 if submit:
                     os.chdir(subd)
                     subprocess.run(["sbatch","jac"+h])
                     os.chdir(topdir)
             continue
         print("Need to run go for "+h)
    return prmtopdone == len(allH)


def main():
    args = parse_args()
    if not Path(args.molecule+".pdb").is_file():
        print("I need "+args.molecule+".pdb in the current directory")
        sys.exit(1)
    listsfile = args.lists

    lists = []
    with open(listsfile, "r") as f:
         lists = f.readlines()

    allH = collectH(lists)
    ensureAllH(allH, args.molecule, args.submit)

if __name__ == "__main__":
    main()
