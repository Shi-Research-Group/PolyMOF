#!/usr/bin/env python3

import argparse
import uuid
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS coeff commands to list a given style"
        "so that the commands can be used with hybrid style"
    )
    parser.add_argument("input_file", help="LAMMPS input file")
    parser.add_argument("command", help="LAMMPS command type (pair, bond, angle, etc")
    parser.add_argument("style", help="LAMMPS style to include with the command")

    args = parser.parse_args()

    if not args.command in ["pair","bond","angle","dihedral","improper"]:
        parser.error("Command must be one of pair, bond, angle, dihedral, improper")

    return args

def make_hybrid_coeff(param_file, default_types):
    temp_file = param_file+"."+uuid.uuid4().hex
    with open(param_file, "r") as fin, open(temp_file, "w") as fout:
        for l in fin:
            bits = l.split()
            found = False
            for c in default_types.keys():
                if len(bits) > 0 and bits[0] == c+"_coeff":
                    pos = 2
                    if c == "pair":
                        pos = 3
                    s = default_types[c]
                    if bits[pos] != s:
                        fout.write(" ".join(bits[:pos])+" "+s+" "+" ".join(bits[pos:])+"\n")
                    else:
                        fout.write(l)
                    found = True
            if not found:
                fout.write(l)

    shutil.move(temp_file, param_file)
                   
                        
def main():
    args = parse_args()
    input_file = args.input_file
    temp_file = input_file+"."+uuid.uuid4().hex
    command = args.command
    command_coeff = command+"_coeff"
    style = args.style
    with open(input_file, 'r') as fin, open(temp_file, 'w') as fout:
        for line in fin:
            if line.startswith(command_coeff):
                # do stuff
                bits = line.split()
                pos = 2
                if command == "pair":
                    pos = 3
                if bits[pos] != style:
                    fout.write(" ".join(bits[:pos])+" "+style+" "+" ".join(bits[pos:])+"\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)

    shutil.move(temp_file, input_file)

if __name__ == "__main__":
    main()
