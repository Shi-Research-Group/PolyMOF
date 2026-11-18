#!/usr/bin/env python

import sys
from pathlib import Path
from polymof_util import read_params
import os
from datetime import datetime
import time
import subprocess
import shutil
from polymof_util import read_params
from prepare_anneal import prepare_anneal
import statistics
from math import sqrt
import polymof_setup
import argparse
from count_bonds import count_bonds
import continue_reaction

parser = argparse.ArgumentParser(
    description="Checks the status of each system, reports results"
)
parser.add_argument("--average", action="store_true", help="Report average densities rather than density from each replica")
parser.add_argument("--setup", action="store_true", help="Run setup where needed")
args = parser.parse_args()

systems = []
with open("systems", "r") as f:
    for l in f:
        bits = l.split()
        if len(bits) == 0 or bits[0] == "#":
            continue
        if len(bits) < 3:
            print("unrecognized line:", l)
            sys.exit(1)
        npoly = int(bits[0])
        size = int(bits[1])
        comp = int(bits[2])
        replicas = 1
        if len(bits) > 3:
            replicas = int(bits[3])
        no3 = len(bits) > 4 and "no3" in bits[3:]
        charge = len(bits) > 4 and "charge" in bits[3:]
        fe3 = len(bits) > 4 and "fe3" in bits[3:]
        dirname = f"s{size}c{comp}r"
        systems.append({"npoly": npoly,
                       "size": size,
                       "comp": comp,
                       "dir": dirname,
                       "replicas": replicas,
                       "no3": no3,
                       "charge": charge,
                       "fe3": fe3})


jobs = subprocess.run(["squeue","-M","all","-u","ajs42","-o","%j %T"], capture_output=True, text=True)
if jobs.returncode != 0:
    print("Could not run squeue, result", jobs.returncode)
    sys.exit(1)
# we will grab header lines, but long as we don't have a job named "CLUSTER:" or "NAME" we'll be fine
jobs = jobs.stdout.splitlines()
newjobs = {}
for j in jobs:
    bits = j.split()
    if len(bits) != 2:
        continue
    newjobs[bits[0]] = bits[1]
jobs = newjobs
def grab_job_name(jfile):
    if not Path(jfile).is_file():
        print("Could not find job file", jfile)
        sys.exit(1)
    with open(jfile, "r") as f:
        for l in f:
            if l.startswith("#SBATCH --job-name="):
                return l[len("#SBATCH --job-name="):].strip()
        print("Could not find job name in", jfile)
    sys.exit(1)

def check_job(jfile, jobs):
    jobname = grab_job_name(jfile)
    if jobname in jobs:
        return jobs[jobname], jobname
    return None, jobname

def write_params(params_file, s):
    npoly = s['npoly']
    nPBImonomers = 120
    nFe = nPBImonomers * s['comp'] // 40
    nIM = nFe * 12

    if s['size'] == 2:
        nPBImonomers *= 2
        nFe *= 2
        nIM *= 2

    nPBI = nPBImonomers // npoly

    if nPBI * npoly != nPBImonomers:
        print(f"Could not find nPBI for {npoly} {s['size']}")
        sys.exit(1)

    if False and s['comp'] == 1 and s['size'] == 1:
        # we bumped monomers from 100 to 120, so this isn't a thing anymore
        # nFe will be 2.5, scale everything up to 3
        nPBI = 24
        nFe = 3
        nIM = 288

    with open(params_file, "w") as f:
        f.write(f"npoly {npoly}\n")
        f.write(f"nPBI {nPBI}\n")
        f.write(f"nFe {nFe}\n")
        f.write(f"nIM {nIM}\n")
        f.write(f"molecule {npoly}mer\n")
        f.write("acpath /projects/academic/kofke/ajs42/PolyMOF/ac/\n")
        if nFe > 0:
            f.write("total_steps 5000000\n")

def init_reaction(s, basedir, reactdir, replica, anneal, no3, charge, fe3):
    shutil.copytree("./Fe.base", reactdir, symlinks=True)
    write_params(reactdir+"/params", s)
    with open(reactdir+"/jlammps", "r") as f:
        contents = f.read()
    jobname = s['dir']+str(replica)
    if anneal:
        jobname += "a"
    if no3:
        jobname += "n"
    if charge:
        jobname += "c"
    if fe3:
        jobname += "f"
    contents = contents.replace("JOBNAME", jobname)
    contents = contents.replace("JOBNAME", jobname)
    lmpfile = "reaction.$n.lmp"
    if anneal:
        lmpfile = "anneal.lmp"
        if s['comp'] == 0:
            lmpfile= "anneal.PBI.lmp"
    contents = contents.replace("LMPFILE", lmpfile)
    with open(reactdir+"/jlammps", "w") as f:
        f.write(contents)

now = time.time()
allj = {"PENDING": [], "RUNNING": []}
topdir = os.getcwd()
for s in systems:
    basedir="Fe."+str(s['npoly'])+"tm"
    if not Path(basedir).is_dir():
        print("skipping ", str(s), "missing", basedir)
        continue

    density_results = []
    for replica in range(1,s['replicas']+1):
        reactdir=basedir+"/reaction."+s['dir']+str(replica)
        if s['comp'] > 0 and not s['no3'] and not s['fe3']:
            if not Path(reactdir).is_dir():
                print(str(s), "missing", reactdir)
                init_reaction(s, basedir, reactdir, replica, False, False, False, False)
                if args.setup:
                    print("running setup for ", reactdir)
                    os.chdir(reactdir)
                    polymof_setup.main()
                    os.chdir(topdir)
                else:
                    print("initialized, please run setup.py")
                    sys.exit(0)
                continue
            if not Path(reactdir+"/log.lammps").is_file() and \
               not Path(reactdir+"/log.lammps.1").is_file():
                jstat, jname = check_job(reactdir+"/jlammps", jobs)
                if jstat == None:
                    if args.setup and (not Path(reactdir+"/mol").is_file() or not Path(reactdir+"/FEX-data.1.lammps").is_file()):
                        print("running setup for ", reactdir)
                        os.chdir(reactdir)
                        polymof_setup.main()
                        os.chdir(topdir)
                    else:
                        print(str(s), "reaction", reactdir, "not started (no log.lammps)")
                else:
                    allj[jstat].append(jname)
                continue
            ndat = Path(reactdir+"/n.dat")
            if not ndat.is_file() or ndat.stat().st_size < 50:
                jstat, jname = check_job(reactdir+"/jlammps", jobs)
                if jstat == None:
                    print(str(s), "reaction", reactdir, "n.dat missing or too small, trying to collect n")
                    os.chdir(reactdir)
                    subprocess.run(["cbn"])
                    os.chdir(topdir)
                else:
                    allj[jstat].append(jname)
                continue
            with open(reactdir+"/n.dat", "rb") as f:
                f.seek(-50, os.SEEK_END)
                lastline = None
                while True:
                    l = f.readline().decode('utf-8')
                    if not l:
                        break
                    lastline = l
            totalsteps = int(lastline.split()[0])
            params = read_params(reactdir+"/params")
            if totalsteps < params['total_steps']:
                loglammps = Path(reactdir+"/log.lammps")
                ndat = Path(reactdir+"/n.dat")
                jstat, jname = check_job(reactdir+"/jlammps", jobs)
                if jstat == None:
                    os.chdir(reactdir)
                    result = continue_reaction.read_reactions()
                    os.chdir(topdir)
                    total_steps_from_input = -1
                    if isinstance(result, tuple):
                        last, last_hours, last_steps, total_steps_from_input = result
                    if loglammps.is_file() and ndat.stat().st_mtime < loglammps.stat().st_mtime:
                        if total_steps_from_input >= params['total_steps']:
                            print(reactdir, "apparently finished, please updating n.dat")
                            os.chdir(reactdir)
                            subprocess.run(["cbn"])
                            os.chdir(topdir)
                        elif total_steps_from_input > 0:
                            print(reactdir, "needs more steps ("+str(total_steps_from_input)+")")
                            if args.setup:
                                os.chdir(reactdir)
                                continue_reaction.main()
                                os.chdir(topdir)
                        else:
                            print(reactdir, "needs more steps ("+str(total_steps_from_input)+")")
                            print("    current status: "+result)
                    elif loglammps.is_file():
                        if args.setup:
                            os.chdir(reactdir)
                            continue_reaction.main()
                            os.chdir(topdir)
                        else:
                            print(reactdir, "needs more steps ("+str(totalsteps)+") needs continue_reaction")
                    else:
                        print(reactdir, "needs more steps ("+str(totalsteps)+") perhaps ready for job submission")
                else:
                    allj[jstat].append(jname)
                continue
        annealdir=basedir+"/anneal."+s['dir']+str(replica)
        if s['no3']:
            if not Path(annealdir).is_dir():
                # if this system is for no3, but annealing without no3 hasn't started, then just ignore for now
                continue
            annealdir += "n"
            if s['charge']:
                annealdir += "c"
        if s['fe3']:
            if not Path(annealdir).is_dir():
                # if this system is for fe3, but fe6 annealing hasn't started, then just ignore for now
                continue
            annealdir += "f"
        if not Path(annealdir).is_dir():
            if s['comp'] == 0:
                print(str(s), "annealing dir missing", annealdir)
                init_reaction(s, basedir, annealdir, replica, True, False, False, False)
                if args.setup:
                    print("running setup for ", annealdir)
                    os.chdir(annealdir)
                    polymof_setup.main()
                    os.chdir(topdir)
                else:
                    print("initialized, please run setup.py")
                    sys.exit(0)
                continue
            else:
                print(str(s), "preparing annealing", annealdir)
                prepare_anneal(reactdir, annealdir, s['no3'], s['charge'], s['fe3'])
                print("submit job", annealdir)
                continue
        if not Path(annealdir+"/log.lammps").is_file():
            jstat, jname = check_job(annealdir+"/jlammps", jobs)
            if jstat == None:
                if s['comp'] == 0 and (not Path(annealdir+"/box.pdb").is_file() or not Path(annealdir+"/data.lammps").is_file()):
                    print("running setup for ", annealdir)
                    os.chdir(annealdir)
                    polymof_setup.main()
                    os.chdir(topdir)
                else:
                    print(str(s)+" annealing not started "+annealdir+"/log.lammps")
            else:
                allj[jstat].append(jname)
            continue
        if not Path(annealdir+"/densityavg.dat").is_file():
            jstat, jname = check_job(annealdir+"/jlammps", jobs)
            if jstat == None:
                print(str(s)+" missing "+annealdir+"/densityavg.dat")
                os.chdir(annealdir)
                subprocess.run(["dtdt"])
                os.chdir(topdir)
            else:
                allj[jstat].append(jname)
            continue
        if not Path(annealdir+"/densityavg.dat").is_file():
            continue
        avg = None
        with open(annealdir+"/densityavg.dat", "r") as f:
            for l in f:
                bits = l.split()
                avg = bits[0]
                err = bits[1]
        if avg is not None:
            density_results.append(avg)
    if len(density_results) > 0:
        no3 = 1 if s['no3'] else 0
        charge = 1 if s['charge'] else 0
        fe3 = 1 if s['fe3'] else 0
        if args.average and len(density_results) > 1:
            density_results = [float(x) for x in density_results]
            avg = statistics.mean(density_results)
            err = statistics.stdev(density_results)
            print(s['npoly'], s['size'], s['comp'], no3, charge, fe3, avg, err/sqrt(len(density_results)-1))
        else:
            print(s['npoly'], s['size'], s['comp'], no3, charge, fe3, *density_results)

if len(allj["PENDING"]) > 0:
    print("pending:", " ".join(allj["PENDING"]))
if len(allj["RUNNING"]) > 0:
    print("running:", " ".join(allj["RUNNING"]))
