#!/usr/bin/env python3
"""Give atoms of one type that have few bonds their own atom type and
parameters.

Atoms of the chosen type (e.g. fe) with at most N bonds are moved to a new
atom type (new label).  Every bond, angle, dihedral and improper that involves
one of those atoms gets a new type whose label uses the new atom label.  The
new types take their parameters from the param file when a line matches the
original type labels (and the kind), and otherwise copy the original
parameters.  Original types are left untouched.

Param file lines:  <kind> label1-label2-... <params...> # (ignored comment)
Parm file lines:   <kind>_coeff N <params...> # label1-label2-...

The data file is rewritten (type counts, labels, masses, pair/bond coeffs,
atoms and the bonds/angles/dihedrals/impropers) and the new coefficient lines
are appended to a copy of the parm file.  Bond, angle and dihedral labels also
match the param file in reverse order.  Any param line that is not used, or
any inconsistency, prints an error and exits with status 1 (nothing written).
"""
import argparse
import re
import sys

KINDS = ("bond", "angle", "dihedral", "improper")
SECTION = {"bond": "Bonds", "angle": "Angles", "dihedral": "Dihedrals", "improper": "Impropers"}
NATOMS = {"bond": 2, "angle": 3, "dihedral": 4, "improper": 4}
COEFF_RE = re.compile(r"^\s*(bond|angle|dihedral|improper)_coeff\s+(\S+)\s+(.*?)\s*#\s*(\S+)\s*$")
PAIR_RE = re.compile(r"^\s*pair_coeff\s+(\d+)\s+(\d+)\s+(.*?)\s*#\s*(\S+)\s*$")
HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z ]*?(\s*#.*)?$")
COUNT_RE = re.compile(r"^(\s*)(\d+)(\s+)(atom|bond|angle|dihedral|improper) types\b(.*)$")
ATOM_TYPE_COL = {"atomic": 1, "charge": 1, "full": 2, "molecular": 2, "bond": 2, "angle": 2}


def fail(errors):
    for e in errors:
        print("ERROR:", e, file=sys.stderr)
    sys.exit(1)


def read_param(path):
    new, errors = {}, []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            body = line.split("#", 1)[0].split()
            if not body:
                continue
            if len(body) < 3 or body[0] not in KINDS:
                errors.append(f"{path}:{n}: cannot parse line: {line.rstrip()}")
                continue
            key = (body[0], body[1])
            if key in new:
                errors.append(f"{path}:{n}: duplicate entry for {body[0]} {body[1]}")
            new[key] = (n, " ".join(body[2:]))
    if errors:
        fail(errors)
    return new


def lookup(param, kind, labels):
    if (kind, labels) in param:
        return (kind, labels)
    if kind != "improper":
        rev = "-".join(reversed(labels.split("-")))
        if (kind, rev) in param:
            return (kind, rev)
    return None


def read_data(path):
    """Return (preamble lines, ordered list of [header, body lines])."""
    with open(path) as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    pre, sections, cur = [], [], None
    for i, line in enumerate(lines):
        if i > 0 and line.strip() and HEADER_RE.match(line.strip()):
            cur = [line, []]
            sections.append(cur)
        elif cur is None:
            pre.append(line)
        else:
            cur[1].append(line)
    for s in sections:
        s[1] = [l for l in s[1] if l.strip()]
    return pre, sections


def find(sections, name):
    for s in sections:
        if s[0].split("#")[0].strip() == name:
            return s
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parm_file", help="LAMMPS parameter file (*_coeff lines)")
    ap.add_argument("param_file", help="file with replacement parameters")
    ap.add_argument("-d", "--data", required=True, help="input LAMMPS data file")
    ap.add_argument("-D", "--data-out", required=True, help="output LAMMPS data file")
    ap.add_argument("-o", "--output", required=True, help="output parameter file")
    ap.add_argument("-t", "--atom-type", required=True,
                    help="atom type (label or number) to update")
    ap.add_argument("-n", "--max-bonds", type=int, required=True,
                    help="only atoms with at most this many bonds are updated")
    ap.add_argument("--new-label", help="label for the new atom type (default: <label>x)")
    args = ap.parse_args()

    errors = []
    param = read_param(args.param_file)
    pre, sections = read_data(args.data)

    counts = {k: 0 for k in ("atom",) + KINDS}
    for l in pre:
        m = COUNT_RE.match(l)
        if m:
            counts[m.group(4)] = int(m.group(2))

    labels = {}
    for kind in ("atom",) + KINDS:
        sec = find(sections, f"{kind.capitalize()} Type Labels")
        labels[kind] = {}
        if sec:
            for l in sec[1]:
                a, b = l.split(None, 1)
                labels[kind][int(a)] = b.strip()
        elif counts[kind]:
            errors.append(f"data file has no {kind.capitalize()} Type Labels section")
    if errors:
        fail(errors)

    t = args.atom_type
    by_label = {v: k for k, v in labels["atom"].items()}
    if t in by_label:
        old_type = by_label[t]
    elif t.isdigit() and int(t) in labels["atom"]:
        old_type = int(t)
    else:
        fail([f"atom type {t} not found in data file"])
    old_label = labels["atom"][old_type]
    new_label = args.new_label or old_label + "x"
    if new_label in by_label:
        fail([f"atom label {new_label} already exists"])
    new_type = counts["atom"] + 1

    atoms_sec = find(sections, "Atoms")
    style = (atoms_sec[0].split("#", 1) + [""])[1].strip() if atoms_sec else ""
    if style not in ATOM_TYPE_COL:
        fail([f"unsupported atom style '{style}' in Atoms header"])
    col = ATOM_TYPE_COL[style]
    atom_type, atom_rows = {}, []
    for l in atoms_sec[1]:
        w = l.split()
        atom_rows.append(w)
        atom_type[int(w[0])] = int(w[col])

    nbonds = {}
    bonds_sec = find(sections, "Bonds")
    for l in (bonds_sec[1] if bonds_sec else []):
        for a in l.split()[2:4]:
            nbonds[int(a)] = nbonds.get(int(a), 0) + 1
    selected = {a for a, ty in atom_type.items()
                if ty == old_type and nbonds.get(a, 0) <= args.max_bonds}
    if not selected:
        fail([f"no atoms of type {old_label} with <= {args.max_bonds} bonds"])
    print(f"{len(selected)} of {sum(1 for v in atom_type.values() if v == old_type)} "
          f"{old_label} atoms have <= {args.max_bonds} bonds", file=sys.stderr)

    with open(args.parm_file) as f:
        parm_lines = f.read().split("\n")
    if parm_lines and parm_lines[-1] == "":
        parm_lines.pop()
    coeff, pair = {}, None
    for l in parm_lines:
        m = COEFF_RE.match(l)
        if m:
            coeff[(m.group(1), int(m.group(2)))] = m.group(3)
        m = PAIR_RE.match(l)
        if m and int(m.group(1)) == old_type and int(m.group(2)) == old_type:
            pair = m.group(3)

    # newtype[kind][(old type, flags)] = new id; flags mark selected atoms
    # in the order of the type label
    newtype = {k: {} for k in KINDS}
    newlabel = {k: {} for k in KINDS}
    rows = {}
    for kind in KINDS:
        sec = find(sections, SECTION[kind])
        rows[kind] = []
        for l in (sec[1] if sec else []):
            w = l.split()
            rows[kind].append(w)
            ids = [int(x) for x in w[2:2 + NATOMS[kind]]]
            flags = tuple(a in selected for a in ids)
            if not any(flags):
                continue
            ty = int(w[1])
            tl = labels[kind].get(ty)
            if tl is None:
                errors.append(f"{kind} type {ty} has no label")
                continue
            inst = [labels["atom"][atom_type[a]] for a in ids]
            if "-".join(inst) == tl:
                fl = flags
            elif "-".join(reversed(inst)) == tl:
                fl = flags[::-1]
            else:
                errors.append(f"{kind} {w[0]}: atom labels {'-'.join(inst)} do not match type label {tl}")
                continue
            key = (ty, fl)
            if key not in newtype[kind]:
                parts = [new_label if f else p for p, f in zip(tl.split("-"), fl)]
                newtype[kind][key] = counts[kind] + len(newtype[kind]) + 1
                newlabel[kind][key] = "-".join(parts)
            w[1] = str(newtype[kind][key])
    if errors:
        fail(errors)

    newcoeff = {k: [] for k in KINDS}
    used = set()
    for kind in KINDS:
        for key, nid in sorted(newtype[kind].items(), key=lambda kv: kv[1]):
            ty = key[0]
            pk = lookup(param, kind, labels[kind][ty])
            if pk:
                used.add(pk)
                p = param[pk][1]
            elif (kind, ty) in coeff:
                p = coeff[(kind, ty)]
            else:
                errors.append(f"no {kind}_coeff for type {ty} ({labels[kind][ty]}) in {args.parm_file}")
                continue
            newcoeff[kind].append((nid, p, newlabel[kind][key]))
    for key, (n, _) in param.items():
        if key not in used:
            errors.append(f"{args.param_file}:{n}: no match for {key[0]} {key[1]} "
                          f"(no such type involves a selected {old_label} atom)")
    if errors:
        fail(errors)

    for w in atom_rows:
        if int(w[0]) in selected:
            w[col] = str(new_type)
    atoms_sec[1] = [" ".join(w) for w in atom_rows]
    for kind in KINDS:
        sec = find(sections, SECTION[kind])
        if sec:
            sec[1] = [" ".join(w) for w in rows[kind]]
        if newtype[kind]:
            find(sections, f"{kind.capitalize()} Type Labels")[1] += \
                [f"{nid} {newlabel[kind][k]}" for k, nid in sorted(newtype[kind].items(), key=lambda kv: kv[1])]
        counts[kind] += len(newtype[kind])
    counts["atom"] += 1
    find(sections, "Atom Type Labels")[1].append(f"{new_type} {new_label}")

    for name, what in (("Masses", "Masses"), ("Pair Coeffs", "Pair Coeffs")):
        sec = find(sections, name)
        if sec:
            val = [l.split(None, 1)[1] for l in sec[1] if int(l.split()[0]) == old_type]
            if not val:
                fail([f"{what} entry for type {old_type} missing in data file"])
            sec[1].append(f"{new_type} {val[0]}")
            if name == "Pair Coeffs" and pair is None:
                pair = val[0]
    if find(sections, "PairIJ Coeffs"):
        fail(["PairIJ Coeffs in data file are not supported"])
    if newcoeff["bond"]:
        bc = find(sections, "Bond Coeffs")
        if bc:
            bc[1] += [f"{nid} {p}" for nid, p, _ in newcoeff["bond"]]

    if pair is None:
        fail([f"no pair coefficients for {old_label} in {args.parm_file} or the data file"])

    out = []
    for l in pre:
        m = COUNT_RE.match(l)
        if m:
            l = f"{m.group(1)}{counts[m.group(4)]}{m.group(3)}{m.group(4)} types{m.group(5)}"
        out.append(l)
    for header, body in sections:
        out += [header, ""] + body + [""]
    with open(args.data_out, "w") as f:
        f.write("\n".join(out) + "\n")

    extra = ["", f"# Added by replace_params.py for {old_label} atoms with <= {args.max_bonds} bonds"]
    extra.append(f"pair_coeff {new_type} {new_type} {pair} # {new_label}")
    for kind in KINDS:
        for nid, p, lab in newcoeff[kind]:
            extra.append(f"{kind}_coeff {nid} {p} # {lab}")
    with open(args.output, "w") as f:
        f.write("\n".join(parm_lines + extra) + "\n")


if __name__ == "__main__":
    main()
