#!/usr/bin/env python

import argparse
import sys
import data2molecule
from operator import itemgetter
import numpy as np

def parse_args():
    def list_of_metals(arg):
        return [m for m in arg.split(",")]

    def list_of_molecules(arg):
        list1 = list(arg.split(","))
        return [m.split(":") for m in list1]

    parser = argparse.ArgumentParser(
            description="Analyze LAMMPS data file to identify complexes, determine the size of each complex and to determine the number of each type of molecule in the complex"
    )
    parser.add_argument("data_file", help="LAMMPS data file to analyze")
    parser.add_argument("--metal", type=list_of_metals, help="Atom type(s) of metal linker in complexes", required=True)
    parser.add_argument("--molecules", help="A molecule name and size to look for in the configuration, specified as name1:size1,name2:size2...", type=list_of_molecules, required=True)
    parser.add_argument("--check-intra", type=int, metavar='NUM_BONDS', help="Check for nearby atoms bonded to the same metal.  If argument is given, atoms are looked for within the given number of bonds.")
    parser.add_argument("--special-atom", help="Print all atoms in molecules that are directly bonded to this metal atom")
    parser.add_argument("--wrap", action='store_true', help="Detect complexes that wrap around periodic bonudaries")
    parser.add_argument("--count-bonds", action='store_true', help="Print how many metal atoms have various numbers of bonds")
    parser.add_argument("--comp", action='store_true', help="Print composition ratios")
    return parser.parse_args()

def walkUntilMetal(ba, atomsByID, bondMap, seenAtoms, metal, boxL):
    """Walk through a molecule starting at atom ba.  Atoms are marked in seenAtoms as we see them;
    we will never see atoms in the seenAtoms that is passed in.  The list of not-metal atoms found
    is returned along with a list of the metals separately."""
    myAtoms = []    # atoms I found
    stackAtoms = [ba] # atoms I found but haven't walked
    metalsFound = {}  # metal atoms I found, bonded to this fragment
    unwrapped_stack_coords = [np.array([float(x) for x in atomsByID[ba][4:7]])]
    while len(stackAtoms) > 0:
        a = stackAtoms.pop()
        ra = unwrapped_stack_coords.pop()

        if atomsByID[a][2] in metal:
            # just remember that we saw the metal
            if a in metalsFound:
                metalsFound[a].append(ra)
            else:
                metalsFound[a] = [ra]
            continue

        if a in seenAtoms:
            continue

        seenAtoms[a] = True
        for b in bondMap[a]:
            stackAtoms.append(b)
            rb = np.array([float(x) for x in atomsByID[b][4:7]])
            if boxL is not None:
                rb = (rb - ra + 0.5*boxL) % boxL - 0.5*boxL + ra
            unwrapped_stack_coords.append(rb)

        myAtoms.append(a)

    return myAtoms, metalsFound

def walkUntilAtom(startAtom, searchAtom, bondMap, atomsByID, metal, maxBonds):
    """Walk through a fragment, starting at startAtom and look for searchAtom.  We will
    only walk for maxBonds.  We return all atoms we find that are bonded to searchAtom."""
    bondsWalked = 0
    originBond = {}
    prevList = [startAtom]
    nextList = []
    seenAtoms = {}
    nearBonds = {}
    while True:

        if len(prevList) == 0:
            if bondsWalked == maxBonds:
                return nearBonds
            prevList = nextList
            nextList = []
            bondsWalked += 1
            originBond = {}
            continue

        a = prevList.pop()

        if atomsByID[a][2] in metal:
            continue

        if a in seenAtoms:
            continue

        seenAtoms[a] = True
        for ba in bondMap[a]:
            if ba == searchAtom and bondsWalked > 0:
                nearBonds[a] = bondsWalked
            else:
                nextList.append(ba)
        if searchAtom == '11' and False:
            print(bondsWalked, nextList)

def doCheckIntra(metalsFound, fragAtoms, bondMap, atomsByID, metal, maxBonds):
    metalCounts = {m: 0 for m in metalsFound}
    for ma in metalsFound:
        metalCounts[ma] += 1
        if metalCounts[ma] == 2:
            pairsFound = {}
            # we saw this metal bonded to this fragment multiple times
            # check if the bonded atoms are nearby (within check_intra bonds)
            for ba in bondMap[ma]:
                if ba not in fragAtoms:
                    continue
                # walk breadth-first from ba.  try to find ma again within check_intra steps
                nearBonds = walkUntilAtom(ba, ma, bondMap, atomsByID, metal, maxBonds)
                if len(nearBonds) > 0:
                    for nearB,numB in nearBonds.items():
                        p = (ba, nearB)
                        if ba > nearB:
                            p = (nearB,ba)
                        pairsFound[p] = numB
            if len(pairsFound) > 0:
                print("metal", ma, "has near bonds", pairsFound)
    #print("metalCounts", metalCounts)

def walkComplexes(atomsByID, metalAtoms, bondMap, metal, moleculeMap, revTypeMap, check_intra=False, special=None, boxL=None):
    seenAtoms = {}
    metalStack = []
    frags = []
    complexEdges = {"m"+m: [] for m in metalAtoms}
    fragMetalCoords = {}
    atomFrag = {}
    if special:
        print("immediate bonds", bondMap[str(special)])
    for ma in metalAtoms:
        seenAtoms[ma] = True
        myComplex = {}
        #print(ma, [revTypeMap[atomsByID[ba][2]] for ba in bondMap[ma]])
        #if len(bondMap[ma]) == 5:
            #print(ma, "has 5 bonds")
        for ba in bondMap[ma]:
            if ba in seenAtoms:
                if ma == special:
                    #print(ba, "already visited")
                    print("\n".join(atomFrag[ba]))
                continue
            fragAtoms, metalsFound = walkUntilMetal(ba, atomsByID, bondMap, seenAtoms, metal, boxL)
            if check_intra:
                doCheckIntra(metalsFound, fragAtoms, bondMap, atomsByID, metal, check_intra)
            for fa in fragAtoms:
                atomFrag[fa] = fragAtoms
            if ma == special:
                #print(ba, "visited")
                print("\n".join(fragAtoms))
            fragSize = len(fragAtoms)
            if fragSize not in moleculeMap:
                print("Found fragment of unexpected size: "+str(fragSize))
            fragID = len(frags)
            if fragSize not in moleculeMap:
                print("fragment has unexpected size:", fragSize)
                print("\n".join(fragAtoms))
                sys.exit(1)
            frags.append(moleculeMap[fragSize])
            fragMetalCoords[fragID] = metalsFound
            complexEdges["f"+str(fragID)] = ["m"+id for id in metalsFound]
            for m in metalsFound:
                complexEdges["m"+m].append("f"+str(fragID))

    complexes = []
    complexes_wrapped = []
    seenNodes = {}
    for ma in metalAtoms:
        metalCoords = {ma: np.array([float(x) for x in atomsByID[ma][4:7]])}

        # walk through complex starting at this metal atom
        mid = "m"+ma
        complexMetals = []
        if mid in seenNodes:
            continue
        numMetals = 1
        numFrags = {m: 0 for m in moleculeMap.values()}
        numFrags["metal"] = 0
        stackNodes = [mid]
        wrapXYZ = [False, False, False]
        while len(stackNodes) > 0:
            n = stackNodes.pop()
            if n in seenNodes:
                continue
            if n[0] == "m":
                numFrags["metal"] += 1
                mid = int(n[1:])
                complexMetals.append(mid)
            else:
                fragID = int(n[1:])
                numFrags[frags[fragID]] += 1
                # grab metals from fragMetalCoords, 
                if boxL is not None:
                    offset = None
                    for mm in fragMetalCoords[fragID]:
                        if mm in metalCoords:
                            offset = metalCoords[mm] - fragMetalCoords[fragID][mm][0]
                            break
                    if offset is None:
                        print("Oops, couldn't find", mm, "in metalCoords", metalCoords.keys())
                        sys.exit(1)
                    for mm in fragMetalCoords[fragID]:
                        newCoord = fragMetalCoords[fragID][mm][0] + offset
                        if mm in metalCoords:
                            wrapped = np.round((newCoord - metalCoords[mm])/boxL)
                            for i in range(3):
                                wrapXYZ[i] = wrapXYZ[i] or (wrapped[i] != 0).item()
                        else:
                            metalCoords[mm] = newCoord
            seenNodes[n] = True
            stackNodes += complexEdges[n]
        if boxL is not None:
            numFrags["wrap"] = wrapXYZ
        numFrags["metals"] = complexMetals
        complexes.append(numFrags)
    kl = ["metal", *list(moleculeMap.values())]
    complexes.sort(key=itemgetter(*kl))
    tmpComplexes = []
    prevC = None
    for c in complexes:
        if c == prevC and boxL is None:
            tmpComplexes[len(tmpComplexes)-1]['num'] += 1
        else:
            tmpComplexes.append({'num': 1, 'molecules': c})
        prevC = c
    return tmpComplexes

def print_bonds(atomsByID, metalAtoms, bondMap):
    bond_counts = {}
    for ma in metalAtoms:
        n = len(bondMap[ma])
        if n not in bond_counts:
            bond_counts[n] = 0
        bond_counts[n] += 1

    print(bond_counts)

def main():
    args = parse_args()
    moleculeList = [row[0] for row in args.molecules]
    moleculeMap = {int(row[1]): row[0] for row in args.molecules}
    #print(moleculeMap)
    data = data2molecule.parse_data_file(args.data_file)
    headers = data[0]
    sections = data[1]
    extra_lines = data[2]
    boxL = None
    if args.wrap:
        for el in extra_lines:
            bits = el.split()
            if bits[2] == "xlo" and bits[0] == "0":
                boxL = np.zeros(3)
                boxL[:] = float(bits[1])
    atomsByID = {row[0]: row for row in sections['Atoms']}
    typeMap = {row[1]: row[0] for row in sections['Atom Type Labels']}
    revTypeMap = {row[0]: row[1] for row in sections['Atom Type Labels']}
    metal = args.metal
    metal = [typeMap[m] for m in metal]
    #moleculeAtoms = {m: [row[0] for row in sections['Atoms'] if row[1] == m] for m in molecules}

    metalAtoms = [row[0] for row in sections['Atoms'] if row[2] in metal]
    #bondMap = {a: [row[3] for row in sections['Bonds'] if row[2]==a] + [row[2] for row in sections['Bonds'] if row[3]==a] for a in atomsByID.keys()}
    bondMap = {a: [] for a in atomsByID.keys()}
    for row in sections['Bonds']:
        a1 = row[2]
        a2 = row[3]
        bondMap[a1].append(a2)
        bondMap[a2].append(a1)
    if args.count_bonds:
        print_bonds(atomsByID, metalAtoms, bondMap)
        return

    complexes = walkComplexes(atomsByID, metalAtoms, bondMap, metal, moleculeMap, revTypeMap, args.check_intra, args.special_atom, boxL)
    if not args.check_intra and not args.special_atom:
        totals = {s: 0 for s in moleculeMap.values()}
        totals["metal"] = 0
        for c in complexes:
            if args.comp and c['molecules'][moleculeList[0]] == 0:
                continue
            if not args.comp:
                print(str(c['num'])+": "+str(c['molecules']))
            for m in c['molecules']:
                if m != "wrap" and m != "metals":
                    totals[m] += c['num']*c['molecules'][m]
        if args.comp:
            print(f"{totals['metal']/totals[moleculeList[0]]/10:3.3f}  {totals[moleculeList[1]]/totals['metal']:3.3f} {totals[moleculeList[1]]/totals[moleculeList[0]]/10:3.3f}")
        else:
            print("Totals:",totals)
    

if __name__ == "__main__":
    main()
