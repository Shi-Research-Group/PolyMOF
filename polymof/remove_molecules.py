#!/usr/bin/env python3
"""
remove_molecules.py

Remove whole molecules from a LAMMPS data file based on the number of atoms
they contain, and write out a new, fully re-numbered data file (atoms, bonds,
angles, dihedrals, impropers, velocities all updated/renumbered consistently).

A "molecule" is identified by the molecule-ID column in the Atoms section
(this requires an atom_style that has a molecule-ID, e.g. "full", "molecular",
"angle", etc.). Any molecule whose atom count matches --natoms or is not more
than --max-atoms is deleted.  Every bond/angle/dihedral/improper/velocity entry
that references one of its atoms is also deleted.

Usage:
    python remove_molecules.py input.data output.data --natoms 12
    python remove_molecules.py input.data output.data --max-atoms 20

Notes:
    * Atom Type Labels / Bond Type Labels / Masses / Pair Coeffs / etc. are
      copied through unchanged since they describe types, not atoms.
    * Atom IDs, and Bond/Angle/Dihedral/Improper IDs are renumbered to be
      contiguous starting at 1, which is required by LAMMPS after atoms are
      removed.
    * Molecule IDs themselves are left as-is (gaps in molecule numbering are
      fine for LAMMPS).
"""

import argparse
import sys
from collections import defaultdict, OrderedDict

# Sections keyed by exact header text (without trailing comment) that contain
# one data row per *atom* and therefore need atom-ID remapping/filtering.
ATOM_INDEXED_SECTIONS = {"Atoms", "Velocities"}

# Sections that reference atom IDs in columns after an entry ID and type ID.
# Maps section name -> number of atom-ID columns that follow (id, type).
TOPOLOGY_SECTIONS = {
    "Bonds": 2,
    "Angles": 3,
    "Dihedrals": 4,
    "Impropers": 4,
}

HEADER_COUNT_KEYS = [
    "atoms", "bonds", "angles", "dihedrals", "impropers",
]

# Section headers that must be copied through verbatim (types/coeffs, not
# per-atom data).
PASSTHROUGH_HINTS = (
    "Type Labels", "Masses", "Coeffs", "BondBond", "BondAngle",
    "MiddleBondTorsion", "EndBondTorsion", "AngleTorsion", "AngleAngleTorsion",
    "BondBond13", "AngleAngle",
)


def parse_data_file(path):
    """Parse a LAMMPS data file into header lines, counts, box lines, and an
    ordered dict of sections: name -> (comment, [raw_lines])."""
    with open(path) as fh:
        lines = fh.readlines()

    header_lines = []
    counts = {}
    box_lines = []
    sections = OrderedDict()

    i = 0
    n = len(lines)

    # First line is the mandatory comment/title line.
    title = lines[0].rstrip("\n")
    i = 1

    # Header block: counts + box bounds, ends right before the first section
    # keyword line (a line whose first non-blank token is alphabetic and is
    # followed only by blank lines / not by numeric header tokens).
    def is_section_header(line):
        stripped = line.strip()
        if not stripped:
            return False
        # header sections start with a capitalized keyword and have no
        # trailing numeric header tokens like "xlo xhi"
        first_word = stripped.split()[0]
        return first_word[0].isalpha() and first_word[0].isupper()

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped == "":
            # peek ahead - if next non-blank is a section header, this blank
            # line ends the header block; otherwise keep it (e.g. the blank
            # line that separates counts from box bounds)
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and is_section_header(lines[j]):
                i += 1
                break
            header_lines.append("")
            i += 1
            continue
        if is_section_header(line):
            break
        header_lines.append(line.rstrip("\n"))
        tokens = stripped.split()
        # counts lines look like "<int> <name...>"
        if len(tokens) >= 2 and tokens[0].lstrip("-").isdigit():
            key = " ".join(tokens[1:])
            counts[key] = int(tokens[0])
        i += 1

    # Now parse sections
    while i < n:
        if lines[i].strip() == "":
            i += 1
            continue
        header_line = lines[i].strip()
        parts = header_line.split("#", 1)
        sec_name = parts[0].strip()
        sec_comment = ("#" + parts[1]) if len(parts) > 1 else ""
        i += 1
        # skip blank line(s) after header
        while i < n and lines[i].strip() == "":
            i += 1
        body = []
        while i < n and lines[i].strip() != "":
            body.append(lines[i].rstrip("\n"))
            i += 1
            # a section ends at the next blank line; but keep consuming
            # consecutive non-blank lines only
        sections[sec_name] = (sec_comment, body)

    return title, header_lines, counts, sections


def get_atom_style(sections):
    if "Atoms" in sections:
        comment = sections["Atoms"][0]
        style = comment.replace("#", "").strip()
        if style:
            return style
    return None


ATOM_STYLE_MOL_COLUMN = {
    # column index (0-based, after the leading atom-id column) of molecule-id
    "full": 1,
    "molecular": 1,
    "angle": 1,
    "bond": 1,
    "atomic": None,
    "charge": None,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Remove molecules with a given atom count from a LAMMPS data file.")
    ap.add_argument("input", help="input LAMMPS data file")
    ap.add_argument("output", help="output LAMMPS data file to write")
    ap.add_argument("--natoms", type=int, default=-1,
                     help="remove every molecule that has exactly this many atoms")
    ap.add_argument("--max-atoms", type=int, default=-1,
                     help="remove every molecule that has no more than this many atoms")
    args = ap.parse_args()

    if args.natoms == -1 and args.max_atoms == -1:
        ap.error("Must specify either --natoms or --max-atoms")
    if args.natoms != -1 and args.max_atoms != -1:
        ap.error("Cannot specify both --natoms and --max-atoms")

    return args

def main():
    args = parse_args()
    remove_molecules(args.input, args.natoms, args.max_atoms, args.output_data)

def remove_molecules(input_data, natoms, max_atoms, output_data):
    title, header_lines, counts, sections = parse_data_file(input_data)

    if "Atoms" not in sections:
        sys.exit("error: input file has no Atoms section")

    atom_style = get_atom_style(sections)
    mol_col = None
    if atom_style is not None:
        mol_col = ATOM_STYLE_MOL_COLUMN.get(atom_style)
    if mol_col is None:
        # fall back: assume "full"-like layout (id, mol, type, ...) since most
        # molecule-oriented data files use it; warn the user.
        mol_col = 1
        sys.stderr.write(
            f"warning: could not determine molecule-ID column for atom_style "
            f"'{atom_style}', assuming column 2 (id mol type ...)\n")

    # --- Parse Atoms section: atom_id -> (mol_id, full_line_tokens) ---
    atom_mol = {}
    atom_lines = {}
    for line in sections["Atoms"][1]:
        tokens = line.split()
        aid = int(tokens[0])
        molid = int(tokens[1 + mol_col - 1]) if False else int(tokens[mol_col])
        atom_mol[aid] = molid
        atom_lines[aid] = line

    # --- Count atoms per molecule ---
    mol_atom_ids = defaultdict(list)
    for aid, molid in atom_mol.items():
        mol_atom_ids[molid].append(aid)

    def should_remove(count):
        return count == natoms or count < max_atoms

    removed_mol_ids = {molid for molid, ids in mol_atom_ids.items()
                        if should_remove(len(ids))}
    removed_atom_ids = set()
    for molid in removed_mol_ids:
        removed_atom_ids.update(mol_atom_ids[molid])

    if not removed_mol_ids:
        sys.stderr.write("warning: no molecules matched the removal criteria; "
                          "output will be identical to input (but renumbered)\n")

    kept_atom_ids = sorted(set(atom_mol) - removed_atom_ids)

    # --- Build old-id -> new-id remap for atoms (contiguous, 1-based) ---
    new_atom_id = {old: new for new, old in enumerate(kept_atom_ids, start=1)}

    # --- Rewrite Atoms section ---
    def remap_atom_line(line):
        tokens = line.split()
        aid = int(tokens[0])
        tokens[0] = str(new_atom_id[aid])
        return " ".join(tokens)

    new_atoms_body = [remap_atom_line(atom_lines[aid]) for aid in kept_atom_ids]
    sections["Atoms"] = (sections["Atoms"][0], new_atoms_body)

    # --- Rewrite Velocities section (same atom-id filter/remap), if present ---
    if "Velocities" in sections:
        new_vel_body = []
        for line in sections["Velocities"][1]:
            tokens = line.split()
            aid = int(tokens[0])
            if aid in new_atom_id:
                tokens[0] = str(new_atom_id[aid])
                new_vel_body.append(" ".join(tokens))
        sections["Velocities"] = (sections["Velocities"][0], new_vel_body)

    # --- Rewrite topology sections: Bonds/Angles/Dihedrals/Impropers ---
    for sec_name, natom_cols in TOPOLOGY_SECTIONS.items():
        if sec_name not in sections:
            continue
        new_body = []
        for line in sections[sec_name][1]:
            tokens = line.split()
            # tokens[0] = entry id, tokens[1] = type id, tokens[2:2+natom_cols] = atom ids
            atom_ids = [int(t) for t in tokens[2:2 + natom_cols]]
            if any(aid in removed_atom_ids for aid in atom_ids):
                continue
            remapped = [str(new_atom_id[aid]) for aid in atom_ids]
            new_tokens = [tokens[0], tokens[1]] + remapped
            new_body.append(" ".join(new_tokens))
        # renumber entry ids contiguously
        renumbered = []
        for new_id, line in enumerate(new_body, start=1):
            tokens = line.split()
            tokens[0] = str(new_id)
            renumbered.append(" ".join(tokens))
        sections[sec_name] = (sections[sec_name][0], renumbered)
        counts[sec_name.lower()] = len(renumbered)

    counts["atoms"] = len(kept_atom_ids)
    if "Velocities" in sections:
        pass  # velocities count isn't tracked separately in header

    # --- Write output ---
    with open(output_data, "w") as out:
        out.write(title + "\n")

        # Re-emit header count lines in original order/wording, using updated
        # counts where applicable, and any box/type-count lines unchanged.
        for hline in header_lines:
            stripped = hline.strip()
            if stripped == "":
                out.write("\n")
                continue
            tokens = stripped.split()
            if len(tokens) >= 2 and tokens[0].lstrip("-").isdigit():
                key = " ".join(tokens[1:])
                if key in counts:
                    out.write(f"{counts[key]} {key}\n")
                    continue
            out.write(hline + "\n")
        out.write("\n")

        for sec_name, (comment, body) in sections.items():
            header = sec_name + (" " + comment if comment else "")
            out.write(header.rstrip() + "\n\n")
            for line in body:
                out.write(line + "\n")
            out.write("\n")

    n_removed_mols = len(removed_mol_ids)
    n_removed_atoms = len(removed_atom_ids)
    print(f"Removed {n_removed_mols} molecule(s), {n_removed_atoms} atom(s).")
    print(f"Wrote {len(kept_atom_ids)} atoms to {output_data}")


if __name__ == "__main__":
    main()
