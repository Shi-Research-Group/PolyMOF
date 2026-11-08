#!/usr/bin/env python3
"""Redistribute charge around central atoms in a LAMMPS data file (overwritten in place).

Charge file format:
  line 1:   <atom type label> <nominal charge>      e.g. "fe 3"
  others:   <number of bonds> <central atom charge> e.g. "4 1"

Each central atom (of the given type) must have charge 0 in the data file. It gets
the specified charge; its bonded neighbors each share (nominal - central) equally.
Only the charge column of the Atoms section is modified.
"""
import argparse
import sys

SECTIONS = {
    'Atoms', 'Bonds', 'Velocities', 'Masses', 'Angles', 'Dihedrals', 'Impropers',
    'Atom Type Labels', 'Bond Type Labels', 'Angle Type Labels', 'Dihedral Type Labels',
    'Improper Type Labels', 'Pair Coeffs', 'PairIJ Coeffs', 'Bond Coeffs', 'Angle Coeffs',
    'Dihedral Coeffs', 'Improper Coeffs', 'BondBond Coeffs', 'BondAngle Coeffs',
    'MiddleBondTorsion Coeffs', 'EndBondTorsion Coeffs', 'AngleTorsion Coeffs',
    'AngleAngleTorsion Coeffs', 'BondBond13 Coeffs', 'AngleAngle Coeffs', 'Ellipsoids',
    'Lines', 'Triangles', 'Bodies',
}


def read_charge_spec(path):
    rows = [l.split('#')[0].split() for l in open(path)]
    rows = [r for r in rows if r]
    if len(rows) < 2 or len(rows[0]) != 2:
        sys.exit("Charge file needs a '<type> <nominal>' line followed by '<nbonds> <charge>' lines")
    table = {}
    for r in rows[1:]:
        if len(r) != 2:
            sys.exit(f"Bad line in charge file: {' '.join(r)}")
        table[int(r[0])] = float(r[1])
    return rows[0][0], float(rows[0][1]), table


def section_range(lines, name):
    """Return (start, end) line indices of the data rows of a section, or None."""
    for i, l in enumerate(lines):
        if l.split('#')[0].strip() == name:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            k = j
            while k < len(lines) and lines[k].strip() and \
                    lines[k].split('#')[0].strip() not in SECTIONS:
                k += 1
            return j, k
    return None


def replace_charge(line, q_new):
    """Replace the 4th token (charge) in an Atoms line, preserving everything else."""
    body, sep, comment = line.partition('#')
    pos = 0
    for n in range(4):
        tok = body.split()[n]
        start = body.index(tok, pos)
        pos = start + len(tok)
    return body[:start] + f"{q_new:.10g}" + body[pos:] + sep + comment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('datafile', help='LAMMPS data file (overwritten)')
    ap.add_argument('chargefile')
    ap.add_argument('--tol', type=float, default=1e-8, help='tolerance for original central charge == 0')
    args = ap.parse_args()
    alter_charges(args.datafile, args.chargefile, tol)

def alter_charges(datafile, chargefile, tol=1e-8):

    ctype, nominal, table = read_charge_spec(chargefile)
    with open(datafile) as f:
        lines = f.read().split('\n')

    atoms = section_range(lines, 'Atoms')
    bonds = section_range(lines, 'Bonds')
    if atoms is None or bonds is None:
        sys.exit("Data file must have Atoms and Bonds sections")

    labels = {}  # label -> numeric id
    lr = section_range(lines, 'Atom Type Labels')
    if lr:
        for l in lines[lr[0]:lr[1]]:
            t = l.split('#')[0].split()
            if len(t) >= 2:
                labels[t[1]] = t[0]
    if ctype in labels:
        type_id = labels[ctype]
    elif ctype.isdigit():
        type_id = ctype
    else:
        sys.exit(f"Atom type '{ctype}' not found in Atom Type Labels")

    # atom_style full: id mol type q x y z ...; type may be a label or a number
    info = {}  # id -> (line index, type id, charge)
    for i in range(*atoms):
        t = lines[i].split('#')[0].split()
        if t:
            info[t[0]] = (i, labels.get(t[2], t[2]), float(t[3]))

    neighbors = {a: [] for a in info}
    for i in range(*bonds):
        t = lines[i].split('#')[0].split()
        if len(t) >= 4:
            neighbors[t[2]].append(t[3])
            neighbors[t[3]].append(t[2])

    centers = [a for a, v in info.items() if v[1] == type_id]
    if not centers:
        sys.exit(f"No atoms of type '{ctype}' found")

    newq = {}
    delta = {}
    for a in centers:
        q = info[a][2]
        if abs(q) > tol:
            sys.exit(f"Error: central atom {a} has nonzero charge {q}")
        nb = neighbors[a]
        if len(nb) not in table:
            sys.exit(f"Error: atom {a} has {len(nb)} bonds, which is not in the charge file")
        newq[a] = table[len(nb)]
        if not nb:
            if abs(nominal - newq[a]) > tol:
                sys.exit(f"Error: atom {a} has no bonded atoms to absorb charge")
            continue
        share = (nominal - newq[a]) / len(nb)
        for b in nb:
            if info[b][1] == type_id:
                sys.exit(f"Error: central atoms {a} and {b} are bonded to each other")
            delta[b] = delta.get(b, 0.0) + share

    for a, q in newq.items():
        lines[info[a][0]] = replace_charge(lines[info[a][0]], q)
    for a, d in delta.items():
        lines[info[a][0]] = replace_charge(lines[info[a][0]], info[a][2] + d)

    with open(datafile, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Updated {len(centers)} '{ctype}' atoms and {len(delta)} bonded atoms")


if __name__ == '__main__':
    main()
