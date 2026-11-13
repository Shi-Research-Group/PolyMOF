#!/usr/bin/env python

import sys
from pathlib import Path
import random
import os
from make_pentamer_ac import ensureAllH
import subprocess
from amber_to_lammps import validate_files, amber2lammps
from go_hybrid_coeff import make_hybrid_coeff
from find_unique_fragments import find_unique_fragments
from make_reactions import readReactants, writeReactions
from data2reaction import d2r_args, data2reaction
import re
import shutil
import glob
from get_mass import get_mass_for_pdb
from polymof_util import read_params

def readTM(params, file):
    tmlist = []
    with open(file, "r") as f:
        for l in f:
            tmlist.append(l.strip())
    params['tm'] = tmlist
    if len(tmlist) != params['nPBI']:
        print("nPBI is", nPBI, "but", file, "has", len(tmlist))
        sys.exit(1)

def makeTM(params, file):
    tmlist = []
    npoly = params['npoly']
    nPBI = params['nPBI']
    xx = 2**(2*npoly)-1
    digits = len(str(xx))
    with open(file, "w") as f:
       for i in range(nPBI):
           x = random.randrange(0, xx)
           x = f"{x:0{digits}d}"
           tmlist.append(x)
           f.write(x+"\n")
    params['tm'] = tmlist

def linkTM(params):
    m = params['molecule']
    ready = True
    for h in params['tm']:
        if not Path(m+h+".pdb").is_file() and not Path(m+h+".pdb").is_symlink():
            target = params['acpath']+m+".tm/"+m+h+"/"+m+h+".pdb"
            if Path(target).is_file():
                os.symlink(target, m+h+".pdb")
            else:
                ready = False
        if not Path(m+h+".prmtop").is_file() and not Path(m+h+".prmtop").is_symlink():
            target = params['acpath']+m+".tm/"+m+h+"/"+m+h+".prmtop"
            if Path(target).is_file() and Path(target).stat().st_size > 0:
                os.symlink(target, m+h+".prmtop")
            else:
                ready = False
    return ready

def handleTM(params):
    if Path("tm.list").is_file():
        readTM(params, "tm.list")
    else:
        makeTM(params, "tm.list")
    ready = linkTM(params)
    if ready:
        return ready
    updateTMList(params)
    return linkTM(params)

def writePack(params):
    if Path("pack.inp").is_file():
        return
    print("making pack.inp")
    molecule = params['molecule']
    nFe = params['nFe']
    nPBI = params['nPBI']
    nIM = params['nIM']
    nNO3 = params['nNO3']
    npoly = params['npoly']
    L0 = max(npoly * 90 / 5, 90)
    if 'L0' in params:
        L0 = params['L0']
    else:
        params['L0'] = L0
    with open("pack.inp", "w") as f:
        f.write(f"""# All the atoms from diferent molecules will be separated at least 2.0
# Anstroms at the solution.

tolerance 2.0

seed -1

# The file type of input and output files is PDB

filetype pdb

# The name of the output file

output box.pdb

pbc {L0} {L0} {L0}\n\n""")

        if nFe > 0:
            f.write(f"""structure FE.pdb
  number {nFe}
end structure\n\n""")
        tmlist = params['tm']
        for tm in tmlist:
            f.write(f"""structure {molecule}{tm}.pdb
  number 1
end structure\n\n""")
        if nIM > 0:
            f.write(f"""structure IM.pdb
  number {nIM}
end structure\n\n""")
        if nNO3 > 0:
            f.write(f"""structure NO3.pdb
  number {nNO3}
end structure\n\n""")

def updateTMList(params):
    masterDir = params['acpath']+params['molecule']+".tm"
    if not Path(masterDir).is_dir():
        print("Please update code to make nominal ",params['molecule'])
        sys.exit(1)
        os.mkdir(masterDir)
    if not Path("ac").is_symlink():
        os.symlink(masterDir, "ac")
    masterList = masterDir+"/lists"
    tmlistfile = os.getcwd()+"/tm.list"
    if Path(masterList).is_file():
        with open(masterList, "r") as f:
            for l in f:
                if tmlistfile in l:
                    return
    with open(masterList, "a") as f:
        f.write(f"{tmlistfile}\n")

def runAC(molecule, charges):
    if not Path(molecule+".mol2").is_file():
        cmdline = f"../ac/bin/antechamber -i {molecule}.pdb -fi pdb -o {molecule}.mol2 -fo mol2 -at gaff2 -pl 30"
        if charges:
            cmdline += " -c bcc"
        subprocess.run(cmdline.split(), check=True)
    if not Path(molecule+".frcmod").is_file():
        cmdline = f"../ac/bin/parmchk2 -i {molecule}.mol2 -s gaff2 -f mol2 -o {molecule}.frcmod"
        subprocess.run(cmdline.split(), check=True)
    prmtopbase = molecule
    if not charges:
        prmtopbase += "z"
    if not Path("leap.in").is_file():
        with open("leap.in", "w") as f:
            f.write(f"""source leaprc.gaff2
mods = loadAmberParams {molecule}.frcmod
TP = loadMol2 {molecule}.mol2
saveAmberParm TP {prmtopbase}.prmtop {molecule}.inpcrd
quit\n""")
    if not Path(prmtopbase+".prmtop").is_file():
        cmdline = "../ac/bin/tleap -s -f leap.in"
        subprocess.run(cmdline.split(), check=True)
           

def make_prmtops(params):
    if not Path("approach").is_file():
        sys.exit(f"I need an approach file to in {params['acpath']+params['molecule']+'.tm'}")
    approach = None
    with open("approach", "r") as f:
        for l in f:
            approach = l.strip()
    if approach == "full":
        topdir = os.getcwd()
        m = params['molecule']
        for h in params['tm']:
            if Path(m+h+"/"+m+h+".prmtop").is_file():
                continue
            if not Path(m+h+"/"+m+h+".mol2").is_file():
                print("Cannot proceed until ac jobs finish for", h)
                sys.exit(0)
            # mol2 exists, we can call runAC to do the rest
            os.chdir(m+h)
            runAC(m+h, False)
            os.chdir(topdir)
            return True
    if approach == "nocharges":
        # create any z.prmtop files that are missing
        topdir = os.getcwd()
        m = params['molecule']
        for h in params['tm']:
            if Path(m+h+"/"+m+h+"z.prmtop").is_file():
                continue
            os.chdir(m+h)
            runAC(m+h, False)
            os.chdir(topdir)
        print("z.prmtop files have been created, submit job to get charges")
        return False
    if approach == "none":
        print("Submit job to get prmtop files or use build_prmtop")
        return False
    print("Unrecognized approach for antechamger", approach)
    sys.exit(1)

def runPack(params):
    writePack(params)
    cmdline = "packmol -i pack.inp"
    subprocess.run(cmdline.split(), check=True)

def a2l(params):
    m = params['molecule']
    topologies = []
    counts = []
    charges = []
    if params['nFe'] > 0:
        topologies += ["FE.prmtop"]
        counts += [params["nFe"]]
        charges += [params['feCharge']]
    topologies += [m+h+".prmtop" for h in params['tm']]
    counts += [1 for i in range(params['nPBI'])]
    charges += [0 for i in range(params['nPBI'])]
    if params['nIM'] > 0:
        topologies += ["IM.prmtop"]
        counts += [params["nIM"]]
        charges += [0]
    if params['nNO3'] > 0:
        topologies += ["NO3.prmtop"]
        counts += [params["nNO3"]]
        charges += [-1]
    validate_files(topologies, counts, "box.pdb")
    rv = amber2lammps("data.lammps", "parm.lammps", topologies, counts, "box.pdb",
                 charges, 3.8, True, False, True)
    if not rv:
        raise

def writeFullReactions():
    reactantLines = find_unique_fragments("data.lammps", ["fe","nd"], 3, False)
    with open("reactions", "w") as f:
        f.write("Reactants\n\n")
        for l in reactantLines:
            f.write(l+"\n")
        f.write("\n")

    # yes, we're just reading back in what we just wrote
    # readReactants processes some of the info that we wrote
    reactants = readReactants("reactions")
    writeReactions("reactions", reactants, 6)

def fix_reactions():
    if not Path("../mol").is_dir():
        print("I do not see a mol directory in my parent directory")
        print("And I am too cowardly to assume I should create my own")
        sys.exit(1)
    if Path("mol").is_dir():
        print("I should not have a real mol subdirectory")
        sys.exit(1)
    if not Path("mol").is_symlink():
        os.symlink("../mol", "mol")

    with open("in.reactions.full.1", "r") as f:
        contents = f.read()

    with open("in.reactions.full.check", "r") as f:
       contents_check = f.read()

    if contents_check != contents:

        contents = re.sub(r"([^ ]*\.(molecule_template|map|))", "mol/\\1", contents)
        # replace generic parameter value placeholders with our specific values
        contents = contents.replace(" ID ", " rxns ")
        contents = contents.replace(" group-ID ", " all ")
        contents = contents.replace(" bond/react ", " bond/react  stabilization yes statted_grp 0.03 ")
        contents = contents.replace("react-group-ID", "all")
        contents = contents.replace("Nevery", "1")
        contents = contents.replace("Rmin", "0")
        contents = contents.replace("Rmax", "3.0")
        with open("in.reactions.full.1", "w") as f:
            f.write(contents)
    
        with open("in.reactions.full.check", "r") as f:
           contents_check = f.read()
        if contents_check != contents:
           print("Unexpected contents for in.reactions.full.1 -- did not match in.reactions.full.check")
           sys.exit(1)

    # all these files should already exist in the mol directory we're re-using
    files = glob.glob("*X_*.molecule_template") + glob.glob("*X_*.map")
    for f in files:
        os.remove(f)

def fix_lmp(file, pdb, params):
    with open(file, "r") as f:
        contents = f.read()

    if "LL1" not in contents and "LLL" not in contents and "RC" not in contents:
        return

    rc = 15
    if 'rc' in params:
        rc = params['rc']
    contents = contents.replace("RC", str(rc))

    targetDensity = 0.6
    if 'density' in params:
        targetDensity = params['density']
    rv = get_mass_for_pdb(pdb)
    d = rv['density'] * 1e24 / 6.02214076e23
    Lnew = rv['L'][0] * (d / targetDensity)**(1/3);

    if 'L0' in params:
        L0 = params['L0']
    else:
        datafile = None
        for l in contents.splitlines():
            if l.startswith("read_data "):
                datafile = l.split()[1]
        with open(datafile, "r") as f:
            for l in f:
                if 'xlo' in l:
                    L0 = float(l.split()[1])
                    break
    if 'LL1' in contents:
        for i in range(1, 5):
            Li = L0 + i * (Lnew - L0) / 4
            contents = contents.replace("LL"+str(i), f"{Li:.15f}")
    elif 'LLL' in contents:
        contents = contents.replace("LLL", f"{Lnew:.15f}")

    with open(file, "w") as f:
        f.write(contents)

    
def main():
    clobber = False
    print("Starting...")
    params = read_params("params")
    print("handling tm")
    ready = handleTM(params)
    if not ready:
        clobber = True
        print("Trying to prepare prmtops for PBI")
        simdir=os.getcwd()
        os.chdir(params['acpath']+params['molecule']+".tm")
        ready = ensureAllH(params['tm'], params['molecule'], False)
        if not ready:
            # try to create prmtop files, bail if need job submission
            ready = make_prmtops(params)
            if not ready:
                return
        os.chdir(simdir)
        ready = linkTM(params)
        if not ready:
           return

    if clobber or not Path("box.pdb").is_file():
        print("Writing pack.inp and running packmol")
        runPack(params)
        clobber = True
    if clobber or not Path("data.lammps").is_file():
        print("Converting to LAMMPS data files")
        a2l(params)
        if params['nFe'] > 0:
            print("Converting to LAMMPS hybrid parameters")
            make_hybrid_coeff("parm.lammps", {"angle": "harmonic", "dihedral": "fourier"})
        clobber = True
    if params['nFe'] > 0 and ( clobber or not Path("reactions").is_file() ):
        print("Finding reactant fragments")
        writeFullReactions()
        clobber = True
    if params['nFe'] > 0 and ( clobber or not Path("FEX-data.1.lammps").is_file() ):
        print("Generating reaction data file")
        d2ra = d2r_args("FEX", "reactions", ["parm.lammps"], ["extra.param"], "FEX-parm.lammps",
                        "data.lammps", "FEX-data.1.lammps", "in.reactions.full.1")
        data2reaction(d2ra)
        clobber = True
    if params['nFe'] > 0:
        fix_reactions()
    if params['nFe'] > 0:
        fix_lmp("reaction.1.lmp", "box.pdb", params)
    else:
        fix_lmp("anneal.PBI.lmp", "box.pdb", params)

    
if __name__ == "__main__":
    main()
