#!/usr/bin/env python3

import argparse
import sys
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reads reactants from a reactions file and then appends a list of reactions to the end."
                    "The first reactant is expected to be the metal center."
                    "The reaction names will match the first letter of the reactant names."
                    "The first letter of the reactant names must be unique."
                    "The updated reactions file can be used by data2reactions.py"
    )
    parser.add_argument("reactions_file", help="reactions file to be read and appended to")
    parser.add_argument("num_reactions", help="Number of reactions in the sequence (coordination number of the metal center")

    args = parser.parse_args()

    return args

def readReactants(reactions_file):
    reactants = []
    firstChars = {}
    with open(reactions_file, 'r') as fin:
        foundReactants = False
        for line in fin:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "Reactions" in line:
                print("Reactions file should not already have reactions")
                sys.exit(1)
            if not foundReactants and "Reactants" in line:
                foundReactants = True
                continue
            if foundReactants:
                # do stuff
                bits = line.split()
                if len(bits) != 3:
                    print("Reactants line format is:")
                    print("NAME data_file initiatorID")
                    print("data_file may be a real data file or a molecule template")
                    sys.exit(1)
                if bits[0][0] in firstChars:
                    print("Fisrt characters of reactants must be unique")
                    sys.exit(1)
                firstChars[bits[0][0]] = True
                reactants.append({"name": bits[0], "file": bits[1], "initiatorID": int(bits[2])})

    if not foundReactants:
        print("Could not find Reactants header in", reactions_file)
        sys.exit(1)

    if len(reactants) == 0:
        print("Found 0 reactants")
        sys.exit(1)

    regexCheck = "".join(firstChars.keys())
    regexCheck = re.compile("^["+regexCheck+"]*$")
    for i, r in enumerate(reactants):
        if len(r['name']) > 1 and regexCheck.match(r['name']):
             print("No reaactant name may be formed from the first letters of any reactant")
             sys.exit(1)

    return reactants

def writeReactions(reactions_file, reactants, num_reactions):
    with open(reactions_file, "a") as f:
        f.write("Reactions\n\n")
        prevProducts = [""]
        for i in range(1,num_reactions+1):
            nextReactants = []
            nn=0
            for j, pp in enumerate(prevProducts):
                if pp:
                    r1 = pp+":"+reactants[0]['name']
                    reactionNameBase = pp
                else:
                    r1 = reactants[0]['name']
                    reactionNameBase = r1[0]
                for k in range(1,len(reactants)):
                    r2 = reactants[k]['name'];
                    reactionName = reactionNameBase + r2[0]
                    f.write(reactionName+" "+r1+" "+r2+"\n")
                    nn += 1
                    keepForNextRound = True
                    for l in range(1,len(reactionNameBase)):
                        if r2[0] < reactionNameBase[l]:
                            keepForNextRound = False
                            break
                    if keepForNextRound:
                        nextReactants.append(reactionName)
            prevProducts = nextReactants
            f.write("\n")

def main():
    args = parse_args()
    reactions_file = args.reactions_file
    num_reactions = int(args.num_reactions)
    reactants = readReactants(reactions_file)
    writeReactions(reactions_file, reactants, num_reactions)

if __name__ == "__main__":
    main()
