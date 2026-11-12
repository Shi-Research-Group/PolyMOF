#!/usr/bin/python

import sys
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(prog='extract-data')

parser.add_argument("-data", required=True)
parser.add_argument("-ufftype", required=True)
parser.add_argument("-gafftype", required=True)
#parser.add_argument("-gaffnum", required=True)
parser.add_argument("-prmtop", action='append', required=True)

args = vars(parser.parse_args())
filename = args['data']
mytypeUFF = args['ufftype']
mytypeGAFF = args['gafftype']
#mytypeNum = int(args['gaffnum'])
infilename = "in."+filename[5:]
prmtopfilename = args['prmtop']

if not Path(filename).exists():
    print("%s must exist" % (filename))
    sys.exit(1)

if not Path(infilename).exists():
    print("%s must exist" % (infilename))
    sys.exit(1)

for i in range(len(prmtopfilename)):
    if not Path(prmtopfilename[i]).exists():
        print("%s must exist" % (prmtopfilename[i]))
        sys.exit(1)

# LAMMPS data file with UFF parameters
mytypeNum = None
imytypeUFF = None
with open(filename, 'r') as fdata:
    section = None
    for line in fdata:
        if line[0] == "\n":
            continue
        if not line[0] == " ":
            line = line.strip()
            section = line
            continue
        bits = line.strip().split()
        if section == "Masses":
            itype = int(bits[0])
            ufftype = bits[3]
            if ufftype == mytypeUFF:
                imytypeUFF = itype
        elif section == "Atoms":
            iatom = int(bits[0])
            iatomtype = int(bits[2])
            if imytypeUFF is None:
                print("imytypeUFF is None")
                sys.exit(1)
            if iatomtype == imytypeUFF:
                mytypeNum = iatom
if imytypeUFF is None:
    print("Could not find imytypeUFF")
    sys.exit(1)
if mytypeNum is None:
    print("Could not determine mytypeNum")
    sys.exit(1)

gafftypes = {}
c = 0
for i in range(len(prmtopfilename)):
    with open(prmtopfilename[i], 'r') as fdata:
        now = False
        for line in fdata:
            line = line.strip()
            if line == "%FLAG AMBER_ATOM_TYPE":
                now = True
                continue
            if now and "%FLAG" in line:
                break
            if not now or "%FORMAT" in line:
                continue;
            bits = line.split()
            for i in range(len(bits)):
                c += 1
                if c == mytypeNum:
                    gafftypes[c] = mytypeGAFF
                    c += 1
                gafftypes[c] = bits[i]
if not mytypeNum in gafftypes:
    gafftypes[mytypeNum] = mytypeGAFF

def parseCoeffs(bits, uffcoeffs):
    itype = int(bits[0])
    parameters = []
    for i in range(1,len(bits)):
        if bits[i] == "#":
            break
        parameters.append(bits[i])
    uffcoeffs[itype] = parameters

def addBondType(section, bits, atomtypes, imytypeUFF, uffdata, gafftypes, bondstyles, gaffBonds):
    nMap = {"bond": 2, "angle": 3, "dihedral": 4, "improper": 4}
    bondtype = section.lower()[:-1]
    n = nMap[bondtype]
    uffatoms = uffdata['atoms']
    uffbonds = uffdata[bondtype+"_coeffs"]
    func = bondstyles[bondtype]
    gaffBonds = gaffBonds[bondtype+"s"]
    found = False
    gt = []
    ut = []
    for i in range(n):
        iatom = int(bits[2+i])
        iatomtype = atomtypes[iatom]
        if iatomtype == imytypeUFF:
            found = True
        ut.append(uffatoms[iatomtype]['uff'])
        gt.append(gafftypes[iatom])
    if not found:
        return
    if bondtype in ["bond","angle","dihedral"] and gt[0] > gt[-1]:
        gt.reverse()
        ut.reverse()
    ibondtype = int(bits[1])
    k = ":".join(gt)
    uffb = uffbonds[ibondtype]
    if func == "hybrid":
       func = uffbonds[ibondtype][0]
       uffb = uffbonds[ibondtype][1:]
    x = {"types": gt,
         "ufftypes": ut,
         "func": func,
         "params": uffb}
    if k in gaffBonds:
        if func == gaffBonds[k]['func'] and uffb == gaffBonds[k]['params']:
            return
        else:
            print("Conflicting assignment for", k)
            print(gaffBonds[k])
            print(x)
            sys.exit(1)
    gaffBonds[k] = x

bondstyles = {}
with open(infilename, 'r') as fdata:
    for line in fdata:
        bits = line.split()
        if len(bits) < 2:
            continue
        if bits[0][-6:] == "_style":
            if bits[1] == "hybrid":
                bondstyles[bits[0][:-6]] = "hybrid"
            else:
                bondstyles[bits[0][:-6]] = " ".join(bits[1:])

imytypeUFF = -1
uffdata = {"atoms": {}, "bond_coeffs": {}, "angle_coeffs": {},
           "dihedral_coeffs": {}, "improper_coeffs": {}, "pair_coeffs": {}}
atomtypes = {}
gaffBonds = {"bonds": {}, "angles": {}, "dihedrals": {}, "impropers": {}}
with open(filename, 'r') as fdata:
    section = None
    for line in fdata:
        if line[0] == "\n":
            continue
        if not line[0] == " ":
            line = line.strip()
            section = line
            continue
        bits = line.strip().split()
        if section == "Masses":
            itype = int(bits[0])
            ufftype = bits[3]
            uffdata["atoms"][itype] = {"uff": ufftype, "mass": float(bits[1])}
            if ufftype == mytypeUFF:
                imytypeUFF = itype
        elif section == "Bond Coeffs":
            parseCoeffs(bits, uffdata["bond_coeffs"])
        elif section == "Angle Coeffs":
            parseCoeffs(bits, uffdata["angle_coeffs"])
        elif section == "Dihedral Coeffs":
            parseCoeffs(bits, uffdata["dihedral_coeffs"])
        elif section == "Improper Coeffs":
            parseCoeffs(bits, uffdata["improper_coeffs"])
        elif section == "Pair Coeffs":
            parseCoeffs(bits, uffdata["pair_coeffs"])
        elif section == "Atoms":
            iatom = int(bits[0])
            iatomtype = int(bits[2])
            atomtypes[iatom] = iatomtype
        elif section in ["Bonds","Angles","Dihedrals","Impropers"]:
            addBondType(section, bits, atomtypes, imytypeUFF, uffdata,
                        gafftypes, bondstyles, gaffBonds)

typedata = uffdata["atoms"][imytypeUFF]
ljdata = uffdata["pair_coeffs"][imytypeUFF]
#print("pair_coeff x x %s %s" % (ljdata[0], ljdata[1]))
#print("%s %s %f 0.000000  lj   %s %s" % (mytypeGAFF, mytypeGAFF, typedata["mass"], ljdata[0], ljdata[1]))
print()
for k in gaffBonds['bonds'].keys():
    kb = gaffBonds['bonds'][k]
    print("bond", "-".join(kb['types']), " ".join(kb['params']), "# ", "(", "-".join(kb['ufftypes']), ")")
    #print(" ".join(kb['types']), kb["func"], " ".join(kb['params']), "# UFF", " ".join(kb['ufftypes']))
print()
for k in gaffBonds['angles'].keys():
    kb = gaffBonds['angles'][k]
    print("angle", "-".join(kb['types']), kb["func"], " ".join(kb['params']), "# ", "(", "-".join(kb['ufftypes']), ")")
    #print(" ".join(kb['types']), kb["func"], " ".join(kb['params']), "# UFF", " ".join(kb['ufftypes']))
print()
for k in gaffBonds['dihedrals'].keys():
    kb = gaffBonds['dihedrals'][k]
    print("dihedral", "-".join(kb['types']), kb["func"], " ".join(kb['params']), "# ", "(", "-".join(kb['ufftypes']), ")")
    #print(" ".join(kb['types']), kb["func"], " ".join(kb['params']), "# UFF", " ".join(kb['ufftypes']))
print()
for k in gaffBonds['impropers'].keys():
    kb = gaffBonds['impropers'][k]
    print("improper", "-".join(kb['types']), kb["func"], " ".join(kb['params']), "# ", "(", "-".join(kb['ufftypes']), ")")
    #print(" ".join(kb['types']), kb["func"], " ".join(kb['params']), "# UFF", " ".join(kb['ufftypes']))
