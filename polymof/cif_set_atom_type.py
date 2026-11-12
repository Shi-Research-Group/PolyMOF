#!/usr/bin/env python

import argparse
import contextlib
import sys

def parse_args():
    def list_of_maps(arg):
        list1 = list(arg.split(","))
        # so ugly
        return {m.split(":")[0]:m.split(":")[1] for m in list1}

    parser = argparse.ArgumentParser(
        description="Force specific atom type(s) for specific atom(s).  Types are added via the CIF atom_site_description field."
    )

    parser.add_argument("cif_file", help="CIF file to update")
    parser.add_argument("--map", help="Atom type map, as x1:T1,x2:T2,x3:T3 where x are atom numbers and T are ataom types", type=list_of_maps, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.cif_file, "r") as f:
        lines = f.readlines()

    with open(args.cif_file, "w") as f:
        atoms = 0
        for l in lines:
            bits = l.split()
            if "_atom_site_description" in l:
                print("_atom_site_description already present")
                atoms = 10
            elif atoms == 0 and "_atom" in l:
                atoms = 1
            elif atoms == 1 and "loop_" in l:
                # all done
                atoms = 3
            elif (atoms == 1 or atoms == 2) and len(bits) == 6:
                if atoms == 1:
                    # first atom, add field
                    f.write("    _atom_site_description\n")
                atoms = 2
                l = l.rstrip("\r\n")
                if bits[1] in args.map:
                    l += "  " + args.map[bits[1]] + "\n"
                else:
                    l += "  ?\n"
            f.write(l)

if __name__ == "__main__":
    main()


