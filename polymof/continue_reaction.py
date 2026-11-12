#!/usr/bin/env python

import sys
from pathlib import Path
import os
from prune_reactions import prune_reactions
from polymof_util import read_params
import shutil
import math
from polymof_util import read_params

def get_last_steps(input_lmp):
    last_steps = 0
    with open(input_lmp, "r") as f:
        for l in f:
            if l.startswith("run "):
                bits = l.split()
                last_steps = int(bits[1])
    return last_steps


def read_reactions():
    last_reaction = last_log = last_data = last_final = 0
    for i in range(1,100):
        if not Path("in.reactions.full."+str(i)).is_file():
            last_reaction = i-1
            break
    if last_reaction < 1:
        print("I can't find any in.reactions.full.* files")
    for i in range(1,100):
        if not Path("log.lammps."+str(i)).is_file():
            last_log = i-1
            break
    if not Path("log.lammps"):
        return "log.lammps must exist"
    for i in range(1,100):
        if not Path("FEX-data."+str(i)+".lammps").is_file():
            last_data = i-1
            break
    for i in range(1,100):
        if not Path("final.data."+str(i)).is_file():
            last_final = i-1
            break
    if last_log+1 != last_reaction or last_reaction != last_data \
          or last_data != last_final:
        return f"last log ({last_log+1}), reaction ({last_reaction}) and FEX-data ({last_data}) and final data ({last_final}) must be the same"

    with open("log.lammps", "rb") as f:
        f.seek(-100, os.SEEK_END)
        lastline = None
        while True:
            l = f.readline().decode('utf-8')
            if not l:
                break
            lastline = l
    if not lastline.startswith("Total wall time:"):
        return "Could not find wall time in log.lammps"
    bits = lastline.split()
    hms = bits[len(bits)-1].split(":")
    # hms will be hours:minutes:seconds even if hours is 0 or hours > 24
    hours = float(hms[0]) + float(hms[1]) / 60 + float(hms[2]) / 3600
    steps = [get_last_steps("reaction."+str(i+1)+".lmp") for i in range(last_reaction)]
    last_steps = steps[-1]
    # not to be confused with params['total_steps'] which is the desired number of steps
    total_steps = sum(steps)

    return last_reaction, hours, last_steps, total_steps

def migrate_reactions(i, last_hours, last_steps, total_steps, params):
    """Creates files for (i+1) reaction simulation from i results.
    Renames log.lammps, updates reactions file, data file and jlammps"""

    shutil.move("log.lammps", "log.lammps."+str(i))

    with open("reaction.NX.lmp", "r") as fnx:
        contents = fnx.read()
    if "KSTEPS" not in contents:
        print("This rection.NX.lmp does not have KSTEPS")
        sys.exit(1)

    with open("final.data."+str(i), "r") as fin, open("FEX-data."+str(i+1)+".lammps", "w") as fout:
        for l in fin:
            if l.startswith("bond_react_"):
                break
            fout.write(l)

    contents = contents.replace("NX", str(i+1))

    target_hours = 12
    if "target_hours" in params:
        target_hours = params['target_hours']
    # assume next sim will be as fast as last sim (it will be faster if we have fewer reactions)
    new_steps = last_steps * target_hours / last_hours
    # now round off to get just one digit
    digits_count = math.floor(math.log10(abs(new_steps)))
    new_steps = round(new_steps, -digits_count)
    # now make sure we don't exceed our total
    if 'total_steps' not in params:
        print("total_steps not present in params")
    elif total_steps + new_steps > params['total_steps']:
        new_steps = params['total_steps'] - total_steps
    if new_steps < 1 or new_steps % 1000 != 0:
        print("new steps", new_steps, "must be positive and a multiple of 1000")
        sys.exit(1)
    print("New sim for", int(new_steps), "steps")
    contents = contents.replace("KSTEPS", str(int(new_steps/1000)))

    with open("reaction."+str(i+1)+".lmp", "w") as fout:
        fout.write(contents)

    prune_reactions("in.reactions.full."+str(i), "FEX-data."+str(i+1)+".lammps",
           "in.reactions.full."+str(i+1), False)

    with open("in.reactions.full."+str(i+1), "r") as f:
        contents = f.read()
    contents = contents.replace(" all 1 0 ", " all 2 0 ")

    with open("in.reactions.full."+str(i+1), "w") as f:
        f.write(contents)

    with open("jlammps", "r") as f:
        contents = f.read()
    if "n="+str(i) not in contents:
        print("Could not find n="+i+" in ljammps!!!!!")
    else:
        contents = contents.replace("n="+str(i), "n="+str(i+1))
        with open("jlammps", "w") as f:
            f.write(contents)

def main():
    params = read_params("params")
    result = read_reactions()
    if isinstance(result, str):
        print(result)
        sys.exit(1)
    last, last_hours, last_steps, total_steps = result
    print(str(last)+" -> "+str(last+1))
    if 'total_steps' in params and total_steps >= params['total_steps']:
        print("Simulation has already completed", total_steps)
        sys.exit(0)
    #print(last, last_hours, last_steps, total_steps)
    migrate_reactions(last, last_hours, last_steps, total_steps, params)

if __name__ == "__main__":
    main()
