#!/usr/bin/env python

import sys
from pathlib import Path
import os
from prune_reactions import prune_reactions
from remove_molecules import remove_molecules
from polymof_util import read_params
import shutil
import math
import argparse
from insert_molecules import im_args, insert_molecules
import re
from replace_params import replace_params, rp_args
from alter_charges import alter_charges

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an annealing simulation directory from a finished reaction directory"
    )
    parser.add_argument("reaction_dir", help="Directory (old) containing reaction simulation")
    parser.add_argument("anneal_dir", help="Directory (new) for anneal simulation")
    parser.add_argument("--no3", action="store_true", help="Add 3x NO3 for each Fe")
    parser.add_argument("--charge", help="Redistribute Fe+3 charge to bonded N based on file contents (see alter_charges.py)")
    parser.add_argument("--fe3", action="store_true", help="Switch UFF parameters from FE6+2 to FE3+2 (octahedral to tetrahedral)")

    args = parser.parse_args()
    if not Path(args.reaction_dir).is_dir():
        parser.error(f"{args.reaction_dir} must exist")
    if Path(args.anneal_dir).is_dir():
        parser.error(f"{args.anneal_dir} must not exist")

    return args

def main():
    args = parse_args()
    prepare_anneal(args.reaction_dir, args.anneal_dir, args.no3, args.charge, args.fe3)

def prepare_anneal(reaction_dir, anneal_dir, no3, charge, fe3):
    os.mkdir(anneal_dir)
    shutil.copy(reaction_dir+"/FEX-parm.lammps", anneal_dir)
    shutil.copy(anneal_dir+"/../../Fe.base/anneal.lmp", anneal_dir)
    finaldata = None
    for i in range(1,100):
        if not Path(reaction_dir+"/final.data."+str(i)).is_file():
           if i == 1:
               print("Could not find any final data in "+reaction_dir)
               sys.exit(1)
           finaldata = reaction_dir+"/final.data."+str(i-1)
           break
    remove_molecules(finaldata, -1, 55, anneal_dir+"/anneal-data.lammps")
    with open(anneal_dir+"/anneal-data.lammps","r") as f:
        lines = f.readlines()

    with open(anneal_dir+"/anneal-data.lammps","w") as f:
        for l in lines:
            if l.startswith("bond_react_props_internal"):
               break
            f.write(l)

    with open(reaction_dir+"/jlammps", "r") as f:
        contents = f.read()
    contents = contents.replace("reaction.$n.lmp", "anneal.lmp")
    job_name_suffix = "a"
    if no3:
        job_name_suffix += "n"
        if charge:
            job_name_suffix += "c"
    if fe3:
        job_name_suffix += "f"
    contents = re.sub(r"--job-name=.*", f"\g<0>{job_name_suffix}", contents)
    with open(anneal_dir+"/jlammps", "w") as f:
        f.write(contents)

    os.symlink("../21step.lmp", anneal_dir+"/21step.lmp")
    os.symlink("../21step0.lmp", anneal_dir+"/21step0.lmp")

    parmfile = anneal_dir+"/FEX-parm.lammps"
    datafile = anneal_dir+"/anneal-data.lammps"
    if no3:
        os.symlink("../../NO3/data-NO3.lammps", anneal_dir+"/data-NO3.lammps")
        os.symlink("../../NO3/parm-NO3.lammps", anneal_dir+"/parm-NO3.lammps")
        params = read_params(reaction_dir+"/params")
        ima = im_args(data=anneal_dir+"/anneal-data.lammps",
                      mol_data=anneal_dir+"/data-NO3.lammps",
                      count=params["nFe"]*3,
                      parm=anneal_dir+"/FEX-parm.lammps",
                      mol_parm=anneal_dir+"/parm-NO3.lammps",
                      out_data=anneal_dir+"/anneal-NO3-data.lammps",
                      out_parm=anneal_dir+"/FEX-NO3-parm.lammps",
                      distance=3,
                      tries=100,
                      seed=None)

        insert_molecules(ima)

        with open(anneal_dir+"/anneal.lmp", "r") as f:
            contents = f.read()

        contents = contents.replace("FEX-parm.lammps", "FEX-NO3-parm.lammps")
        contents = contents.replace("anneal-data.lammps", "anneal-NO3-data.lammps")
            
        lines = contents.split("\n")

        with open(anneal_dir+"/anneal.lmp", "w") as f:
            for l in lines:
                # if we are doing charge (below), then we will update the fe charge in the data file
                # if not, then we need to update it in the input script
                if "21step0.lmp" in l and not charge:
                    f.write("set type fe charge 3\n\n")
                f.write(l+"\n")
        datafile = anneal_dir+"/anneal-NO3-data.lammps"
        parmfile = anneal_dir+"/FEX-NO3-parm.lammps"

        if charge:
            job_name_suffix += "c"
            alter_charges(datafile, "fe.charge")

    if fe3:
        # replace_params adds an angle potential with fourier and doesn't fix the input
        with open(anneal_dir+"/anneal.lmp", "r") as f:
            lines = f.readlines()

        with open(anneal_dir+"/anneal.lmp", "w") as f:
            for l in lines:
                if fe3 and l.startswith("angle_style"):
                    l = l.replace("cosine/periodic", "cosine/periodic fourier")
                f.write(l)

        rpa = rp_args(parmfile, "FE3-extra.param", datafile,
                       datafile, parmfile, "fe", 4, "fe3")
        replace_params(rpa)


if __name__ == "__main__":
    main()
