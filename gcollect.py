#!/usr/bin/env python

from pathlib import Path
import sys
import os
import subprocess
from polymof_util import get_jobs, grab_job_name, check_job
from extract_charge import extract_mol2_charges
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check on all the Gaussian runs that are needed, try to set up or run an additional work, and extract whatever has finished."
    )
    parser.add_argument("--check", action="store_true", help="Check status, but do not write any new files or submit jobs")
    return parser.parse_args()

def read_clusterXYZ(fxyz):
    with open(fxyz,"r") as f:
        return f.readlines()[2:]

def makeGJF(d, name, i, charge, spin, card, last_xyz):
    with open(d+"/base.gjf", "r") as fbase:
        contents = fbase.read()

    contents = contents.replace("NAME", name)
    contents = contents.replace("CARD", card)
    contents = contents.replace("CHARGESPIN", str(charge)+" "+str(spin))
    coords = ""
    for l in last_xyz:
        if isinstance(l, str):
            coords += l
        else:
            coords += (" ".join(l))+"\n"
    contents = contents.replace("COORDS", coords)
    with open(d+"/"+name+".gjf", "w") as fout:
        fout.write(contents)

def make_job(d, name, i, spin):
    with open(d+"/jBase", "r") as fbase:
        contents = fbase.read()

    contents = contents.replace("NAME", name)
    with open(d+"/j"+name, "w") as fout:
        fout.write(contents)

args = parse_args()

jobs = get_jobs()

topdir=os.getcwd()
for i in range(7):
    charge = 0
    if i == 0:
        d = "IM"
        spins = [1,2,3,4,5,6]
        basis = "6-31G(d)"
    else:
        charge = 3
        d = "FeIM"+str(i)
        spins = [2,4,6]
        basis = "genecp"
    if not Path(d).is_dir():
        continue
    Espin = {j: {} for j in range(1,5)}
    for s in spins:
        last_xyz = read_clusterXYZ(d+"/init.xyz")
        incomplete = []
        for j in range(1,5):
            fname = d+"S"+str(s)+"_"+str(j)
            flog = fname+".log"
            fgjf = fname+".gjf"
            method = "B3LYP" if j <= 2 else "MP2"
            card = f"#P {method}/{basis} Scf=XQC"
            if j % 2 == 1:
                card += " Opt Freq"
            else:
                card += " Pop=(ReadRadii,MK) IOp(6/33=2)"
            jprev = j-1
            if j == 3:
                jprev = 1
            if jprev in incomplete or 1 in incomplete:
                break
            if not Path(d+"/"+fgjf).is_file():
                print(d+"/"+fgjf+" does not exist")
                if not args.check:
                    makeGJF(d, fname, i, charge, s, card, last_xyz)
                    make_job(d, fname, i, s)
                    os.chdir(d)
                    subprocess.run(["sbatch","j"+fname])
                    os.chdir(topdir)
                    # we've done what we can for this j
                break
            with open(d+"/"+fgjf, "r") as f:
                flines = f.readlines()
                expected0 = f"%chk={d}S{s}_{j}.chk"
                if flines[0].strip() != expected0:
                    print(fgjf, "First line (chk) is off")
                    print(f"Expected: {expected0}")
                    print(f"found: {flines[0]}")
                    sys.exit(1)
                foundcard = any(flines[il].strip() == card for il in range(10))
                if not foundcard:
                    print("Could not find card ", card, "in", d+"/"+fgjf)
                    sys.exit(1)
                foundchargespin = False
                chargespin_candidates = []
                ichargespin = -1
                for il in range(15):
                    bits = flines[il].split()
                    if len(bits) == 0:
                        continue
                    if bits[0] == "FE":
                        print("Could not find charge spin line in", fgjf)
                        sys.exit(1)
                    if len(bits) == 2 and (bits[0] == "3" or bits[0] == "0") and bits[1] == str(s):
                        foundchargespin=True
                        ichargespin = il
                        break
                    elif len(bits) == 2:
                        chargespin_candidates.append(flines[il])
                if not foundchargespin:
                    print("Could not find chargespin line", fgjf)
                    if len(chargespin_candidates) > 0:
                        print("I did find:\n", chargespin_candidates)
                    sys.exit(1)

                if j == 1:
                    for il in range(1+9*i):
                        if flines[ichargespin+1+il] != last_xyz[il]:
                            print(d, fgjf, "1 coords don't match, line ", il)
                            print("init: ", last_xyz[il])
                            print("gjf:  ", flines[ichargespin+1+il])
                            break
                else:
                    # coords in gjf should match opt coords from previous run
                    for il in range(1,1+9*i+1):
                        bits = flines[ichargespin+il].split()
                        if any(bits[1+dim] != last_xyz[il-1][1+dim] for dim in range(3)):
                            print(il," atom from ", fgjf, "does not match opt from", jprev)
                            print("opt: "+(" ".join(last_xyz[il])))
                            print("gjf: "+(" ".join(bits[1:])))
                            sys.exit(1)
                            break

            jstat = None
            if Path(d+"/j"+fname).is_file():
                jstat, jname = check_job(d+"/j"+fname, jobs)
            elif args.check:
                print("?!?!? I want to create job file for", fname)
            else:
                print("?!?!? Creating job file for", fname)
                make_job(d, fname, i, s)
                jstat = "Created"
            if not Path(d+"/"+flog).is_file():
                if jstat == None:
                    print(d+"/"+flog+" does not exist")
                    print("   !job not running!")
                elif jstat == "RUNNING":
                    print(d+"/"+flog+" does not exist")
                    print("   !job is running!")
                incomplete.append(j)
                continue


            # odd j
            # geometry opt, grab opt config
            # look for Optimization completed, Stationary point found
            # config will be printed 2x after that (Standard orientation)
            # but both configs are the same.  take the first
            # even j, just grab energy
            last_xyz_new = []
            E = None
            with open(d+"/"+flog, "r") as f:
                optcomplete = 0
                countdown = -1
                lastline = None
                for l in f:
                    lastline = l
                    if optcomplete == 0 and "Optimization completed." in l:
                        optcomplete = 1
                    elif optcomplete == 1 and "-- Stationary point found." in l:
                        optcomplete = 2
                    elif optcomplete == 2 and "Standard orientation:" in l:
                        optcomplete = 3
                        countdown = 4
                    elif optcomplete == 3 and countdown > 0:
                        countdown -= 1
                    elif optcomplete == 3 and countdown == 0:
                        if "-------" in l:
                            optcomplete = 4
                            continue
                        bits = l.split()
                        lxyz = last_xyz[len(last_xyz_new)]
                        if isinstance(lxyz, str):
                            el = lxyz.split()[0]
                        else:
                            el = lxyz[0]
                        last_xyz_new.append([el,*bits[3:6]])
                    elif "SCF Done:" in l:
                        # SCF Done will appear twice and we want the second
                        E = float(l.split()[4])
                inc = False
                if not "Normal termination of Gaussian 16" in lastline:
                    inc = True
                    E = None
                    incomplete.append(j)
                if j%2 == 1 and len(last_xyz_new) == 0:
                    jstat, jname = check_job(d+"/j"+fname, jobs)
                    if jstat == 'RUNNING':
                        continue
                    print("Could not find optimized coordinates in", flog)
                    if jstat == 'PENDING':
                        print("  job pending")
                    else:
                        print("   run appears to have failed")
                    break
                if E is None and not inc:
                    print("Oops, could not find energy for optimized geometry", flog)
                    incomplete.append(j)
                elif inc:
                    print(flog, "incomplete")
                Espin[j][s] = E
                if j%2 == 1:
                    last_xyz = last_xyz_new

    for j in [2,4]:
        print(i,j,Espin[j])
        if len(Espin[j]) > 0:
            bestS = None
            minE = None
            for s in Espin[j]:
                E = Espin[j][s]
                if E is None:
                    continue
                E = float(E)
                fname = d+"S"+str(s)+"_"+str(j)
                with open(d+"/"+fname+".E", "w") as f:
                    f.write(str(E)+"\n")
                if minE is None:
                    minE = E
                    bestS = s
                elif E < minE:
                    minE = E
                    bestS = s
            if bestS is not None:
                fmol2 = d+"/"+d+"_"+str(j)+".mol2"
                if Path(fmol2).is_file():
                    mol2time = Path(fmol2).stat().st_mtime
                    flog = d+"/"+d+"S"+str(bestS)+"_"+str(j)+".log"
                    logtime = Path(flog).stat().st_mtime
                    if mol2time > logtime:
                        continue
                # invoke antechamber to produce mol2
                flog = d+"/"+d+"S"+str(bestS)+"_"+str(j)+".log"
                cmdline = f"/user/ajs42/s/ambertools26/bin/antechamber -i {flog} -fi gout -o {fmol2} -fo mol2 -c resp -rn MOL"
                print("Invoking antechamber to get RESP charges for", flog)
                subprocess.run(cmdline.split(), check=True)
                # extract charge from mol2 -> NAME.charge
                charges = extract_mol2_charges(fmol2)
                print(charges)
                feCharge = None
                if i > 0:
                    feCharge = charges[0]
                    charges = charges[1:]
                if len(charges) != 9:
                    print("Expected 9 charges, but found", len(charges), "from", fmol2)
                    sys.exit(1)
                with open(d+"/"+d+"_"+str(j)+".charge", "w") as f:
                    for ic in range(9):
                        f.write(str(ic+1)+" "+charges[ic]+"\n")
                if i > 0:
                    with open(d+"/"+d+"_"+str(j)+".charge.fe", "w") as f:
                        f.write(feCharge+"\n")

            
                

        
