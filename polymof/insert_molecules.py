#!/usr/bin/env python3
"""Insert copies of a single-molecule LAMMPS data file into a main LAMMPS data
file at random positions/orientations, and merge the two parameter files.

Types are matched by type label (Atom/Bond/Angle/Dihedral/Improper Type
Labels): labels already in the main file are reused, new labels get new type
numbers. Coefficients for new types are taken from the molecule parameter
file (renumbered) and appended to a copy of the main parameter file.

Notes:
  * Only orthogonal boxes are supported; periodic boundaries are assumed.
  * Pair/Bond/... Coeffs sections of the input data files are NOT written
    (the merged parameter file carries the coefficients instead).
  * New atoms get zero velocities if the main file has a Velocities section.

Example:
  insert_molecules.py anneal-data.lammps data-NO3.lammps 20 \
      --parm parm.lammps --mol-parm parm-NO3.lammps \
      -o merged-data.lammps --out-parm merged-parm.lammps
"""
import argparse
import re
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

KINDS = ["Atom", "Bond", "Angle", "Dihedral", "Improper"]
COUNT_KEY = {"Atom": "atoms", "Bond": "bonds", "Angle": "angles",
             "Dihedral": "dihedrals", "Improper": "impropers"}
TYPE_KEY = {k: COUNT_KEY[k][:-1] + " types" for k in KINDS}
TOPO = {"Bond": ("Bonds", 2), "Angle": ("Angles", 3),
        "Dihedral": ("Dihedrals", 4), "Improper": ("Impropers", 4)}
COEFF = {"pair_coeff": None, "bond_coeff": "Bond", "angle_coeff": "Angle",
         "dihedral_coeff": "Dihedral", "improper_coeff": "Improper"}
SECTIONS = {"Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals",
            "Impropers", "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs",
            "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",
            "BondBond Coeffs", "BondAngle Coeffs", "MiddleBondTorsion Coeffs",
            "EndBondTorsion Coeffs", "AngleTorsion Coeffs",
            "AngleAngleTorsion Coeffs", "BondBond13 Coeffs", "AngleAngle Coeffs",
            "Atom Type Labels", "Bond Type Labels", "Angle Type Labels",
            "Dihedral Type Labels", "Improper Type Labels"}


def read_data(path):
    with open(path) as f:
        lines = f.read().splitlines()
    d = {"header": {}, "box": {}, "sections": {}, "title": lines[0] if lines else ""}
    sec = None
    for raw in lines[1:]:
        line = raw.split("#")[0].strip()
        name = raw.split("#")[0].strip()
        if name in SECTIONS:
            sec = name
            d["sections"][sec] = []
            continue
        if not line:
            continue
        if sec is None:
            m = re.match(r"^(\S+)\s+(\S+)\s+([xyz]lo)\s+([xyz]hi)$", line)
            if m:
                d["box"][m.group(3)[0]] = (float(m.group(1)), float(m.group(2)))
                continue
            if re.search(r"\bxy\s+xz\s+yz\b", line):
                sys.exit(f"{path}: triclinic boxes are not supported")
            m = re.match(r"^(\d+)\s+(.+)$", line)
            if m:
                d["header"][m.group(2).strip()] = int(m.group(1))
                continue
            sys.exit(f"{path}: cannot parse header line: {raw!r}")
        d["sections"][sec].append(line.split())
    return d


def labels(d, kind):
    sec = d["sections"].get(f"{kind} Type Labels")
    n = d["header"].get(TYPE_KEY[kind], 0)
    if n and not sec:
        sys.exit(f"{kind} types exist but no '{kind} Type Labels' section; "
                 "type labels are required in both data files")
    return {int(r[0]): r[1] for r in (sec or [])}


def build_maps(main, mol):
    """Return per-kind {mol type -> merged type} and the new labels added."""
    maps, new = {}, {}
    for kind in KINDS:
        ml, nl = labels(main, kind), labels(mol, kind)
        rev = {v: k for k, v in ml.items()}
        nxt = max(ml, default=0) + 1
        maps[kind], new[kind] = {}, {}
        for t in sorted(nl):
            if nl[t] in rev:
                maps[kind][t] = rev[nl[t]]
            else:
                maps[kind][t] = nxt
                new[kind][nxt] = nl[t]
                rev[nl[t]] = nxt
                nxt += 1
    return maps, new


def rand_rot(rng):
    return Rotation.from_quat(rng.normal(size=4))


def insert(main, mol, n, cutoff, tries, seed):
    rng = np.random.default_rng(seed)
    lo = np.array([main["box"][a][0] for a in "xyz"])
    L = np.array([main["box"][a][1] for a in "xyz"]) - lo

    matoms = main["sections"]["Atoms"]
    pos = np.array([[float(x) for x in r[4:7]] for r in matoms])
    placed = [((pos - lo) % L)]
    tree = cKDTree(placed[0], boxsize=L)

    natoms = np.array([[float(x) for x in r[4:7]] for r in mol["sections"]["Atoms"]])
    com = natoms.mean(axis=0)
    rel = natoms - com
    new_pos = []
    for i in range(n):
        for attempt in range(tries):
            R = rand_rot(rng)
            center = lo + rng.random(3) * L
            xyz = R.apply(rel) + center
            wrapped = (xyz - lo) % L
            hits = tree.query_ball_point(wrapped, cutoff)
            if not any(len(h) for h in hits):
                break
        else:
            return None, i
        new_pos.append(xyz)
        placed.append(wrapped)
        tree = cKDTree(np.vstack(placed), boxsize=L)
    return new_pos, n


def write_data(path, main, mol, maps, new, new_pos, lo, L):
    ms, ns = main["sections"], mol["sections"]
    n = len(new_pos)
    na_main = len(ms["Atoms"])
    max_id = max(int(r[0]) for r in ms["Atoms"])
    max_mol = max(int(r[1]) for r in ms["Atoms"])
    matoms = ns["Atoms"]
    id_order = {int(r[0]): i for i, r in enumerate(matoms)}

    hdr = dict(main["header"])
    nm = {k: len(ns.get(TOPO[k][0], [])) for k in TOPO}
    hdr["atoms"] += n * len(matoms)
    for k in TOPO:
        hdr[COUNT_KEY[k]] = hdr.get(COUNT_KEY[k], 0) + n * nm[k]
    for k in KINDS:
        hdr[TYPE_KEY[k]] = hdr.get(TYPE_KEY[k], 0) + len(new[k])

    out = []
    w = out.append
    w(main["title"] + f" (+{n} inserted molecules)")
    w("")
    for k in KINDS:
        key = COUNT_KEY[k]
        if key in hdr or k == "Atom":
            w(f"{hdr.get(key, 0)} {key}")
    for k in KINDS:
        key = TYPE_KEY[k]
        if key in hdr:
            w(f"{hdr[key]} {key}")
    w("")
    for a in "xyz":
        w(f"{main['box'][a][0]} {main['box'][a][1]} {a}lo {a}hi")

    for k in KINDS:
        sec = f"{k} Type Labels"
        if sec in ms or new[k]:
            w("")
            w(sec)
            w("")
            for r in ms.get(sec, []):
                w(" ".join(r))
            for t, lab in new[k].items():
                w(f"{t} {lab}")

    w("")
    w("Masses")
    w("")
    for r in ms["Masses"]:
        w(" ".join(r))
    for t, lab in new["Atom"].items():
        old = next(r[0] for r in ns["Masses"] if maps["Atom"][int(r[0])] == t)
        mass = next(r[1] for r in ns["Masses"] if r[0] == old)
        w(f"{t} {mass}")

    atoms_lines, vel_lines = [], []
    for i, xyz in enumerate(new_pos):
        wrapped = (xyz - lo) % L + lo
        img = np.floor((xyz - lo) / L).astype(int)
        for j, r in enumerate(matoms):
            aid = max_id + i * len(matoms) + j + 1
            t = maps["Atom"][int(r[2])]
            im = " ".join(str(x) for x in img[j])
            atoms_lines.append(
                f"{aid} {max_mol + i + 1} {t} {r[3]} "
                f"{wrapped[j][0]:.6f} {wrapped[j][1]:.6f} {wrapped[j][2]:.6f} {im}")
            vel_lines.append(f"{aid} 0.0 0.0 0.0")

    w("")
    w("Atoms # full")
    w("")
    for r in ms["Atoms"]:
        w(" ".join(r))
    out.extend(atoms_lines)

    if "Velocities" in ms:
        w("")
        w("Velocities")
        w("")
        for r in ms["Velocities"]:
            w(" ".join(r))
        out.extend(vel_lines)

    for k, (sec, na) in TOPO.items():
        if hdr.get(COUNT_KEY[k], 0) == 0:
            continue
        w("")
        w(sec)
        w("")
        rows = ms.get(sec, [])
        for r in rows:
            w(" ".join(r))
        cnt = max((int(r[0]) for r in rows), default=0)
        for i in range(n):
            for r in ns.get(sec, []):
                cnt += 1
                ids = [max_id + i * len(matoms) + id_order[int(a)] + 1
                       for a in r[2:2 + na]]
                w(f"{cnt} {maps[k][int(r[1])]} " + " ".join(map(str, ids)))
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def merge_parms(main_path, mol_path, maps, out_path):
    with open(main_path) as f:
        text = f.read()
    if not text.endswith("\n"):
        text += "\n"
    added = []
    with open(mol_path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tok = s.split()
            if tok[0] not in COEFF:
                added.append(s)  # e.g. pair_modify; pass through for review
                continue
            kind = COEFF[tok[0]] or "Atom"
            try:
                if tok[0] == "pair_coeff":
                    i, j = int(tok[1]), int(tok[2])
                    ids, nid = [maps["Atom"][i], maps["Atom"][j]], 2
                else:
                    ids, nid = [maps[kind][int(tok[1])]], 1
            except (ValueError, KeyError):
                sys.exit(f"cannot map types in parameter line: {s}")
            if all(t in orig_ids[kind] for t in ids):
                continue  # already defined by main parameter file
            added.append(" ".join([tok[0]] + [str(t) for t in ids] + tok[1 + nid:]))
    with open(out_path, "w") as f:
        f.write(text)
        f.write("\n# Parameters merged from " + mol_path + "\n")
        f.write("\n".join(added) + "\n")

class im_args:
    def __init__(self, data, mol_data, count, parm, mol_parm, out_data, out_parm, distance, tries, seed):
        self.data = data
        self.mol_data = mol_data
        self.count = count
        self.parm = parm
        self.mol_parm = mol_parm
        self.output = out_data
        self.out_parm = out_parm
        self.distance = distance
        self.tries = tries
        self.seed = seed

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="main LAMMPS data file")
    p.add_argument("mol_data", help="data file containing a single molecule")
    p.add_argument("count", type=int, help="number of molecules to insert")
    p.add_argument("--parm", required=True, help="parameter file for main data")
    p.add_argument("--mol-parm", required=True, help="parameter file for molecule")
    p.add_argument("-o", "--output", default="inserted.data")
    p.add_argument("--out-parm", default="inserted.parm")
    p.add_argument("-d", "--distance", type=float, default=3.0,
                   help="minimum distance to any existing atom (default 3)")
    p.add_argument("-t", "--tries", type=int, default=100,
                   help="attempts per molecule (default 100)")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def main():

    args = parse_args()
    insert_molecules(args)

def insert_molecules(a):
    global orig_ids

    main_d, mol_d = read_data(a.data), read_data(a.mol_data)
    maps, new = build_maps(main_d, mol_d)

    # Types already known to the main file (everything not newly created)
    orig_ids = {}
    for k in KINDS:
        orig_ids[k] = set(range(1, main_d["header"].get(TYPE_KEY[k], 0) + 1))

    new_pos, done = insert(main_d, mol_d, a.count, a.distance, a.tries, a.seed)
    if new_pos is None:
        sys.exit(f"FAILED: could not place molecule {done + 1} of {a.count} "
                 f"after {a.tries} attempts with distance {a.distance}; "
                 "no output written. Reduce the count/distance or raise --tries.")

    lo = np.array([main_d["box"][x][0] for x in "xyz"])
    L = np.array([main_d["box"][x][1] for x in "xyz"]) - lo
    write_data(a.output, main_d, mol_d, maps, new, new_pos, lo, L)
    merge_parms(a.parm, a.mol_parm, maps, a.out_parm)
    print(f"Inserted {a.count} molecules -> {a.output}, {a.out_parm}")


if __name__ == "__main__":
    main()
