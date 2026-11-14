#!/usr/bin/env python3
"""
combine_pdbs.py - Combine atoms from multiple PDB files into a single PDB file.

- Reads ATOM and HETATM records from each input file.
- Writes all atoms out as ATOM records in the combined output, renumbered
  sequentially starting at 1.
- Combines CONECT records from all input files, renumbering the referenced
  atom serial numbers to match the new numbering.
- Writes a CRYST1 record for a rectangular (90/90/90, P 1) box large enough
  that no atom is within min_dist of any box face; atoms are shifted so the
  box origin is at one corner.
- All other record types (HEADER, REMARK, SEQRES, etc.) are dropped.
- Molecules 2, 3, ... are translated along the X axis so that none of their
  atoms lie within a minimum distance (default 5 Angstrom) of any atom
  belonging to an earlier molecule in the combined structure. This is done
  by separating each molecule's bounding sphere (centroid + max atom radius)
  from every previously placed molecule's bounding sphere by at least
  r_i + r_j + min_dist, which guarantees the true point-to-point separation
  is at least min_dist.

Usage:
    python3 combine_pdbs.py -o combined.pdb mol1.pdb mol2.pdb mol3.pdb
    python3 combine_pdbs.py -o combined.pdb --min-dist 5.0 mol1.pdb mol2.pdb
"""

import argparse
import sys
import numpy as np


def parse_pdb(path):
    """Parse ATOM/HETATM and CONECT records from a PDB file.

    Returns:
        atoms: list of dicts with keys: serial, name, altLoc, resName,
               chainID, resSeq, iCode, x, y, z, occupancy, tempFactor,
               element, charge
        conects: list of tuples (serial, [bonded_serial, ...]) using the
                 ORIGINAL serial numbers from this file
    """
    atoms = []
    conects = []
    with open(path) as fh:
        for line in fh:
            record = line[0:6].strip()
            if record in ("ATOM", "HETATM"):
                try:
                    serial = int(line[6:11])
                except ValueError:
                    continue
                name = line[12:16]
                altLoc = line[16:17]
                resName = line[17:20].strip()
                chainID = line[21:22]
                resSeq = line[22:26]
                iCode = line[26:27]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = line[54:60].strip() or "1.00"
                tempFactor = line[60:66].strip() or "0.00"
                element = line[76:78].strip()
                if not element:
                    # best-effort guess from the atom name
                    element = name.strip().rstrip("0123456789")[:2].strip()
                charge = line[78:80].strip() if len(line) >= 80 else ""
                atoms.append(dict(
                    serial=serial, name=name, altLoc=altLoc,
                    resName=resName, chainID=chainID, resSeq=resSeq,
                    iCode=iCode, x=x, y=y, z=z, occupancy=occupancy,
                    tempFactor=tempFactor, element=element, charge=charge,
                ))
            elif record == "CONECT":
                fields = line[6:].split()
                nums = []
                for f in fields:
                    try:
                        nums.append(int(f))
                    except ValueError:
                        pass
                if nums:
                    conects.append((nums[0], nums[1:]))
    return atoms, conects


def bounding_sphere(coords):
    """Return (centroid, radius) such that every point in coords lies
    within `radius` of `centroid`."""
    centroid = coords.mean(axis=0)
    radius = np.linalg.norm(coords - centroid, axis=1).max() if len(coords) else 0.0
    return centroid, radius


def compute_placements(molecules, min_dist):
    """Given a list of coordinate arrays (one per molecule), compute a
    translation vector for each molecule so that bounding spheres are
    separated by at least r_i + r_j + min_dist for all i < j.

    Molecules are placed sequentially along the +X axis.
    """
    translations = []
    placed_centers = []  # (center, radius) after translation
    cursor_x = 0.0
    for coords in molecules:
        centroid, radius = bounding_sphere(coords)
        if not placed_centers:
            new_center = np.array([0.0, 0.0, 0.0])
        else:
            # Ensure separation from every previously placed molecule.
            # Since molecules are placed with non-decreasing x, it is
            # sufficient to satisfy the constraint against the most
            # recently placed one when placing strictly increasing along
            # a single axis (the required x-offset is monotonic).
            required_x = cursor_x
            for (pc, pr) in placed_centers:
                need = pc[0] + pr + radius + min_dist
                if need > required_x:
                    required_x = need
            new_center = np.array([required_x, 0.0, 0.0])
        translation = new_center - centroid
        translations.append(translation)
        placed_centers.append((new_center, radius))
        cursor_x = new_center[0] + radius + min_dist
    return translations


def format_atom_line(serial, atom):
    name = atom["name"]
    # PDB atom name field alignment: element left-justified in cols 13-14
    # unless it's a 4-char name already.
    if len(name.strip()) < 4 and len(atom["element"].strip()) < 2:
        name_fmt = " {:<3}".format(name.strip())
    else:
        name_fmt = "{:<4}".format(name.strip())[:4]
    line = (
        "ATOM  {serial:>5} {name:<4}{altLoc:1}{resName:<3} {chainID:1}"
        "{resSeq:>4}{iCode:1}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:>6}{temp:>6}"
        "          {element:>2}{charge:<2}"
    ).format(
        serial=serial,
        name=name_fmt,
        altLoc=atom["altLoc"] if atom["altLoc"].strip() else " ",
        resName=atom["resName"][:3],
        chainID=atom["chainID"] if atom["chainID"].strip() else " ",
        resSeq=atom["resSeq"].strip() or "1",
        iCode=atom["iCode"] if atom["iCode"].strip() else " ",
        x=atom["x"], y=atom["y"], z=atom["z"],
        occ=atom["occupancy"],
        temp=atom["tempFactor"],
        element=atom["element"],
        charge=atom["charge"],
    )
    return line


def format_conect_lines(serial, bonded):
    """CONECT allows up to 4 bonded atoms per line; split into multiple
    lines if needed."""
    lines = []
    for i in range(0, len(bonded), 4):
        chunk = bonded[i:i + 4]
        line = "CONECT" + "{:>5}".format(serial)
        for b in chunk:
            line += "{:>5}".format(b)
        lines.append(line)
    return lines


CHAIN_IDS = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + \
            [chr(c) for c in range(ord('0'), ord('9') + 1)]


def parse_endpoint(token, n_molecules):
    """Parse a bond endpoint of the form 'MOLECULE:SERIAL', e.g. '2:15',
    meaning atom serial 15 as it appeared in the 2nd input file on the
    command line (1-based molecule index). Returns (mol_idx0, serial)."""
    token = token.strip()
    if ":" not in token:
        raise ValueError(
            f"Bond endpoint '{token}' must be of the form MOLECULE:SERIAL "
            f"(e.g. '1:23'), where MOLECULE is the 1-based index of the "
            f"input file the atom serial number came from."
        )
    mol_str, serial_str = token.split(":", 1)
    try:
        mol_idx = int(mol_str) - 1
        serial = int(serial_str)
    except ValueError:
        raise ValueError(f"Could not parse bond endpoint '{token}' as MOLECULE:SERIAL")
    if not (0 <= mol_idx < n_molecules):
        raise ValueError(
            f"Bond endpoint '{token}' refers to molecule {mol_idx + 1}, "
            f"but only {n_molecules} input file(s) were given"
        )
    return mol_idx, serial


def read_bonds_file(path, n_molecules):
    """Read extra bond specifications from a text file. Each non-blank,
    non-comment line contains two whitespace-separated MOLECULE:SERIAL
    endpoints, e.g.:

        1:23 2:5
        2:5  2:9
    """
    bonds = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected two MOLECULE:SERIAL endpoints, "
                    f"got: {line!r}"
                )
            bonds.append((parse_endpoint(parts[0], n_molecules),
                           parse_endpoint(parts[1], n_molecules)))
    return bonds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Input PDB files, in order")
    ap.add_argument("-o", "--output", required=True, help="Output PDB file")
    ap.add_argument("--min-dist", type=float, default=5.0,
                     help="Minimum allowed distance between atoms of "
                          "different molecules (default: 5.0)")
    ap.add_argument("--no-rechain", action="store_true",
                     help="Keep original chain IDs instead of assigning "
                          "one chain letter per input molecule")
    ap.add_argument("--bond", action="append", default=[],
                     metavar="MOL1:SERIAL1,MOL2:SERIAL2",
                     help="Add an extra bond (CONECT record) between two "
                          "atoms, identified by the 1-based index of the "
                          "input file they came from and their original "
                          "atom serial number in that file, given as a "
                          "single comma-separated argument, e.g. "
                          "'--bond 1:23,2:5' bonds atom 23 of the 1st "
                          "input file to atom 5 of the 2nd input file. "
                          "May be given multiple times.")
    ap.add_argument("--bonds-file",
                     help="Path to a text file with one extra bond per "
                          "line, each line containing two whitespace "
                          "separated MOLECULE:SERIAL endpoints (see "
                          "--bond); '#' starts a comment")
    args = ap.parse_args()

    all_atoms = []   # list of atom-list per molecule
    all_conects = []  # list of conect-list per molecule (original serials)
    for path in args.inputs:
        atoms, conects = parse_pdb(path)
        if not atoms:
            print(f"Warning: no ATOM/HETATM records found in {path}", file=sys.stderr)
        all_atoms.append(atoms)
        all_conects.append(conects)

    coord_arrays = [
        np.array([[a["x"], a["y"], a["z"]] for a in atoms]) if atoms else np.zeros((0, 3))
        for atoms in all_atoms
    ]

    translations = compute_placements(coord_arrays, args.min_dist)

    # Rectangular box: shift everything so the minimum corner of the atoms
    # sits at (min_dist, min_dist, min_dist), and make the box extend
    # min_dist beyond the maximum corner, so no atom is within min_dist of
    # any box face.
    placed = [c + t for c, t in zip(coord_arrays, translations) if len(c)]
    if placed:
        allpts = np.vstack(placed)
        lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    else:
        lo = hi = np.zeros(3)
    box_shift = args.min_dist - lo
    translations = [t + box_shift for t in translations]
    box = (hi - lo) + 2 * args.min_dist

    # Collect requested extra bonds up front so we can fail fast on bad
    # input before doing any work.
    n_molecules = len(args.inputs)
    extra_bond_specs = []
    for spec in args.bond:
        parts = spec.split(",")
        if len(parts) != 2:
            raise SystemExit(
                f"Error: --bond value '{spec}' must contain exactly two "
                f"comma-separated MOLECULE:SERIAL endpoints, e.g. "
                f"'1:23,2:5'"
            )
        a, b = parts
        extra_bond_specs.append(
            (parse_endpoint(a, n_molecules), parse_endpoint(b, n_molecules))
        )
    if args.bonds_file:
        extra_bond_specs.extend(read_bonds_file(args.bonds_file, n_molecules))

    out_atom_lines = []
    out_conect_lines = []
    new_serial = 1
    # (mol_idx, original_serial) -> new_serial, across ALL molecules, used
    # to resolve user-specified extra bonds after the main renumbering pass.
    global_serial_map = {}
    for mol_idx, (atoms, conects, translation) in enumerate(zip(all_atoms, all_conects, translations)):
        serial_map = {}
        chain_id = CHAIN_IDS[mol_idx % len(CHAIN_IDS)]
        for atom in atoms:
            shifted = dict(atom)
            shifted["x"] = atom["x"] + translation[0]
            shifted["y"] = atom["y"] + translation[1]
            shifted["z"] = atom["z"] + translation[2]
            if not args.no_rechain:
                shifted["chainID"] = chain_id
            out_atom_lines.append(format_atom_line(new_serial, shifted))
            serial_map[atom["serial"]] = new_serial
            global_serial_map[(mol_idx, atom["serial"])] = new_serial
            new_serial += 1

        for old_serial, bonded in conects:
            if old_serial not in serial_map:
                continue
            new_bonded = [serial_map[b] for b in bonded if b in serial_map]
            if new_bonded:
                out_conect_lines.extend(
                    format_conect_lines(serial_map[old_serial], new_bonded)
                )

    if extra_bond_specs:
        extra_bonds = {}
        for (mol_a, serial_a), (mol_b, serial_b) in extra_bond_specs:
            key_a, key_b = (mol_a, serial_a), (mol_b, serial_b)
            if key_a not in global_serial_map:
                raise SystemExit(
                    f"Error: extra bond references molecule {mol_a + 1} "
                    f"atom serial {serial_a}, which does not exist"
                )
            if key_b not in global_serial_map:
                raise SystemExit(
                    f"Error: extra bond references molecule {mol_b + 1} "
                    f"atom serial {serial_b}, which does not exist"
                )
            new_a = global_serial_map[key_a]
            new_b = global_serial_map[key_b]
            extra_bonds.setdefault(new_a, []).append(new_b)
            extra_bonds.setdefault(new_b, []).append(new_a)
        for serial in sorted(extra_bonds):
            out_conect_lines.extend(format_conect_lines(serial, extra_bonds[serial]))

    with open(args.output, "w") as out:
        out.write("CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} {:<11}{:>4}\n"
                  .format(box[0], box[1], box[2], 90.0, 90.0, 90.0, "P 1", 1))
        for line in out_atom_lines:
            out.write(line + "\n")
        out.write("TER\n")
        for line in out_conect_lines:
            out.write(line + "\n")
        out.write("END\n")

    n_atoms = new_serial - 1
    print(f"Wrote {n_atoms} atoms and {len(out_conect_lines)} CONECT lines "
          f"from {len(args.inputs)} molecule(s) to {args.output}")


if __name__ == "__main__":
    main()
