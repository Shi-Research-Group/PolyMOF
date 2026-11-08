#!/usr/bin/env python3
"""
data2molecule.py

Convert one or more LAMMPS data files (atom_style full) and/or LAMMPS
molecule template files (e.g. a post-reaction template produced by
`react_molecule.py`) into a single LAMMPS molecule template file
suitable for use with `create_atoms ... mol` / `fix gcmc` /
`fix deposit` / molecule-based inserts, etc.

Each input file is assumed to describe a single fragment (e.g. one
AMBER residue, or an existing molecule template). Input format (LAMMPS
data file vs. molecule template) is auto-detected per file by looking
for an "Atoms" section (data file) vs. a "Coords" section (molecule
template). When multiple files are given, their atoms/bonds/angles/
dihedrals/impropers are concatenated into one molecule, with atom IDs
and atom/bond/angle/dihedral/improper type numbers offset so fragments
don't collide (e.g. type 1 in fragment A and type 1 in fragment B
become distinct types in the merged molecule). A molecule template
fragment's atom/bond/angle/dihedral/improper type *labels* (e.g.
"nc-cc") are preserved and kept distinct from other fragments'
labels/types the same way.

Usage:
    python3 data2molecule.py data.lammps molecule.lammps
    python3 data2molecule.py IM-data.lammps Zn-data.lammps molecule.lammps

    # Use an existing molecule template (e.g. a post-reaction template
    # produced by react_molecule.py) as one of the input fragments,
    # instead of a LAMMPS data file:
    python3 data2molecule.py ZN-IM-post.molecule_template Zn-data.lammps \
        molecule.lammps

    # Keep only atoms within 3 bonds (default) of atom 5 (in data.lammps's
    # own numbering), e.g. to build a small fragment template for
    # `fix bond/react`:
    python3 data2molecule.py --center-atom 5 data.lammps molecule.lammps

    # Multiple data files: repeat --center-atom once per file, in its own
    # (original, pre-merge) numbering, and filter both to within 2 bonds:
    python3 data2molecule.py --center-atom 5 --center-atom 3 --max-bonds 2 \
        IM-data.lammps Zn-data.lammps molecule.lammps

    # A single data file containing multiple distinct molecules (e.g. a Zn
    # ion and imidazole rings sharing one file): select just the atoms
    # belonging to molecule-ID 95 (matched against the 2nd/"mol" column of
    # the Atoms section):
    python3 data2molecule.py --molecule-id 95 ZN-IM-data.lammps im.lammps
"""

import sys
import re
import argparse
from collections import deque

# we only need some of these, but need the rest to know when a section ends
SECTION_HEADERS = {
    "Masses", "Atoms", "Velocities", "Bonds", "Angles",
    "Dihedrals", "Impropers", "Pair Coeffs", "Bond Coeffs",
    "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",
    "BondBond Coeffs", "BondAngle Coeffs", "MiddleBondTorsion Coeffs",
    "EndBondTorsion Coeffs", "AngleTorsion Coeffs",
    "AngleAngleTorsion Coeffs", "BondBond13 Coeffs", "AngleAngle Coeffs",
    "Atom Type Labels", "Bond Type Labels", "Angle Type Labels",
    "Dihedral Type Labels", "Improper Type Labels",
    # per-atom fix property/atom sections LAMMPS appends after a
    # fix bond/react run (e.g. via its own stabilization/instance
    # tracking).
    # include sections here so data files written after such a run
    # still parse correctly instead of having their rows swallowed
    # into whatever section preceded them.
    "bond_react_props_internal", "bond_react_stabilization_internal",
}

# maps the "kind" used throughout this script to the data-file section
# name holding that kind's type -> label mapping
LABEL_SECTIONS = {
    "atom": "Atom Type Labels",
    "bond": "Bond Type Labels",
    "angle": "Angle Type Labels",
    "dihedral": "Dihedral Type Labels",
    "improper": "Improper Type Labels",
}

# Section headers recognized in LAMMPS molecule template files (as
# produced e.g. by react_molecule.py or LAMMPS's own `write_data`-style
# molecule dumps). Only Coords/Types/Charges/Bonds/Angles/Dihedrals/
# Impropers are actually used by this script; the rest are recognized
# just so they're skipped correctly instead of confusing the parser.
MOL_TEMPLATE_SECTION_HEADERS = {
    "Coords", "Types", "Molecules", "Fragments", "Charges", "Diameters",
    "Masses", "Dipoles", "Velocities", "Bonds", "Angles", "Dihedrals",
    "Impropers", "Special Bond Counts", "Special Bonds", "Shake Flags",
    "Shake Atoms", "Shake Bond Types",
}


def extract_label_map(sections, kind):
    """Return a {type_number: label} dict for the given kind, built from
    the corresponding '<Kind> Type Labels' section, if present.
    """
    section = LABEL_SECTIONS[kind]
    return {int(row[0]): row[1] for row in sections.get(section, [])}

COUNT_RE = re.compile(
    r"^\s*(\d+)\s+(atoms|bonds|angles|dihedrals|impropers|"
    r"atom types|bond types|angle types|dihedral types|improper types)\s*$"
)


# return empty string for comment line, other returns full line
def clean_line(line):
    return line.split("#", 1)[0].strip()


KNOWN_EDGE_IDS_RE = re.compile(r"^\s*#\s*Known EdgeIDs:\s*(.*)$")


def read_known_edge_ids(path):
    """Scan a file for a `# Known EdgeIDs: ...` comment (as written by
    react_molecule.py's --map-file support, or by this script's own
    write_molecule_file below) and return the set of atom IDs it lists,
    in that file's own numbering, or an empty set if none is present.

    This lets EdgeIDs recorded by an earlier data2molecule.py
    --map-file extraction (and carried forward by react_molecule.py
    into a post-reaction template) survive a later re-extraction (e.g.
    a --center-atom/--max-bonds filter centered elsewhere) even for
    atoms whose own topology wasn't truncated by that later filter.
    """
    edge_ids = set()
    with open(path) as f:
        for line in f:
            m = KNOWN_EDGE_IDS_RE.match(line)
            if m:
                edge_ids.update(int(tok) for tok in m.group(1).split())
    return edge_ids


def parse_data_file(path):
    counts = {}
    sections = {}
    extra_lines = []
    lines = []

    with open(path) as f:
        for line in f:
            l = clean_line(line)
            if l:
                lines.append(l)

    i = 0
    n = len(lines)

    # Header: counts + box bounds, until first blank line followed by a
    # recognized section keyword.
    while i < n:
        line = lines[i]
        if line in SECTION_HEADERS:
            break
        m = COUNT_RE.match(line)
        if m:
            counts[m.group(2)] = int(m.group(1))
        elif line:
            extra_lines.append(line)
        i += 1

    # Sections
    while i < n:
        header_line = lines[i]
        if header_line in SECTION_HEADERS:
            section = header_line
            i += 1
            rows = []
            while i < n:
                content = lines[i]
                if content in SECTION_HEADERS:
                    break
                rows.append(content.split())
                i += 1
            sections[section] = rows
        else:
            i += 1

    sections["_known_edge_ids"] = read_known_edge_ids(path)
    return counts, sections, extra_lines


def detect_file_format(path):
    """Return "data" or "molecule" depending on whether `path` looks like
    a LAMMPS data file (has an "Atoms" section) or a LAMMPS molecule
    template (has a "Coords" section).
    """
    with open(path) as f:
        for line in f:
            l = clean_line(line)
            if l == "Coords":
                return "molecule"
            if l == "Atoms":
                return "data"
    raise ValueError(
        f"Could not determine file format of {path}: found neither an "
        "'Atoms' section (LAMMPS data file) nor a 'Coords' section "
        "(LAMMPS molecule template)."
    )


def parse_molecule_template(path):
    """Parse a LAMMPS molecule template file (e.g. a post-reaction
    template produced by react_molecule.py) into the same (counts,
    sections) representation used for LAMMPS data files, so it can be
    filtered/merged/written by the exact same code path as a data file
    fragment.

    Coords/Types/Charges are combined into a synthetic "Atoms" section
    (mol ID set to 1 for every atom, since molecule templates don't have
    a mol column -- they describe a single molecule). Each
    Bonds/Angles/Dihedrals/Impropers row's type *label* (e.g. "nc-cc")
    is turned into a per-fragment type number, with the label preserved
    in a synthetic "<Kind> Type Labels" section, mirroring how data
    files' own Type Labels sections work -- so the existing label-aware
    merge/write code handles molecule templates for free.
    """
    lines = []
    with open(path) as f:
        for line in f:
            l = clean_line(line)
            if l:
                lines.append(l)

    i, n = 0, len(lines)

    # Header: counts, until the first recognized section keyword.
    while i < n:
        if lines[i] in MOL_TEMPLATE_SECTION_HEADERS:
            break
        i += 1

    raw_sections = {}
    while i < n:
        header_line = lines[i]
        if header_line in MOL_TEMPLATE_SECTION_HEADERS:
            section = header_line
            i += 1
            rows = []
            while i < n:
                content = lines[i]
                if content in MOL_TEMPLATE_SECTION_HEADERS:
                    break
                rows.append(content.split())
                i += 1
            raw_sections[section] = rows
        else:
            i += 1

    coords = {row[0]: row[1:4] for row in raw_sections.get("Coords", [])}
    types = {row[0]: row[1] for row in raw_sections.get("Types", [])}
    charges = {row[0]: row[1] for row in raw_sections.get("Charges", [])}

    if not coords:
        raise ValueError(f"No 'Coords' section found in molecule template {path}.")

    # Assign a per-fragment integer atom type to each distinct label, in
    # order of first appearance, and record the label mapping (mimicking
    # a data file's "Atom Type Labels" section).
    atom_type_num = {}
    atom_labels = {}
    atoms = []
    for aid in sorted(coords, key=int):
        label = types.get(aid, "1")
        if label not in atom_type_num:
            atom_type_num[label] = len(atom_type_num) + 1
            atom_labels[atom_type_num[label]] = label
        x, y, z = coords[aid]
        q = charges.get(aid, "0.0")
        # id mol type q x y z, matching a data file's Atoms section.
        atoms.append([aid, "1", str(atom_type_num[label]), q, x, y, z])

    sections = {
        "Atoms": atoms,
        "Atom Type Labels": [[str(num), label] for num, label in atom_labels.items()],
    }

    counts = {
        "atoms": len(atoms),
        "atom types": len(atom_type_num),
    }

    for section, kind, natoms in (
        ("Bonds", "bond", 2),
        ("Angles", "angle", 3),
        ("Dihedrals", "dihedral", 4),
        ("Impropers", "improper", 4),
    ):
        type_num = {}
        labels = {}
        rows = []
        for row in raw_sections.get(section, []):
            rid, label = row[0], row[1]
            atom_ids = row[2:2 + natoms]
            if label not in type_num:
                type_num[label] = len(type_num) + 1
                labels[type_num[label]] = label
            rows.append([rid, str(type_num[label]), *atom_ids])
        sections[section] = rows
        if labels:
            sections[LABEL_SECTIONS[kind]] = [
                [str(num), label] for num, label in labels.items()
            ]
        counts[section.lower()] = len(rows)
        counts[f"{kind} types"] = len(type_num)

    sections["_known_edge_ids"] = read_known_edge_ids(path)
    return counts, sections


def parse_fragment_file(path):
    """Parse `path` as either a LAMMPS data file or a LAMMPS molecule
    template (auto-detected), returning a (counts, sections) tuple in
    the common representation used throughout this script.
    """
    fmt = detect_file_format(path)
    if fmt == "molecule":
        return parse_molecule_template(path)
    return parse_data_file(path)


def merge_fragments(fragments):
    """Concatenate parsed (counts, sections) tuples from multiple data
    files into a single (counts, sections) pair, offsetting atom IDs and
    atom/bond/angle/dihedral/improper type numbers so fragments don't
    collide.
    """
    merged = {"Atoms": [], "Bonds": [], "Angles": [], "Dihedrals": [], "Impropers": []}
    merged_labels = {"atom": {}, "bond": {}, "angle": {}, "dihedral": {}, "improper": {}}

    atom_offset = 0
    type_offset = {"atom": 0, "bond": 0, "angle": 0, "dihedral": 0, "improper": 0}
    known_edge_ids = set()

    for counts, sections in fragments:
        atoms = sections.get("Atoms", [])
        known_edge_ids.update(
            aid + atom_offset for aid in sections.get("_known_edge_ids", set())
        )

        for kind in merged_labels:
            for type_num, label in extract_label_map(sections, kind).items():
                merged_labels[kind][type_num + type_offset[kind]] = label

        # id mol type q x y z [nx ny nz]
        for row in atoms:
            new_row = list(row)
            new_row[0] = str(int(row[0]) + atom_offset)
            new_row[2] = str(int(row[2]) + type_offset["atom"])
            merged["Atoms"].append(new_row)

        for section, kind, natoms in (
            ("Bonds", "bond", 2),
            ("Angles", "angle", 3),
            ("Dihedrals", "dihedral", 4),
            ("Impropers", "improper", 4),
        ):
            for row in sections.get(section, []):
                new_row = list(row)
                new_row[0] = str(len(merged[section]) + 1)
                new_row[1] = str(int(row[1]) + type_offset[kind])
                for k in range(2, 2 + natoms):
                    new_row[k] = str(int(row[k]) + atom_offset)
                merged[section].append(new_row)

        atom_offset += len(atoms)
        for kind, label in (
            ("atom", "atom types"),
            ("bond", "bond types"),
            ("angle", "angle types"),
            ("dihedral", "dihedral types"),
            ("improper", "improper types"),
        ):
            type_offset[kind] += counts.get(label, 0)

    merged_counts = {
        "atoms": len(merged["Atoms"]),
        "bonds": len(merged["Bonds"]),
        "angles": len(merged["Angles"]),
        "dihedrals": len(merged["Dihedrals"]),
        "impropers": len(merged["Impropers"]),
    }
    merged["_known_edge_ids"] = known_edge_ids
    return merged_counts, merged, merged_labels


def filter_by_molecule_id(counts, sections, molecule_id):
    """Keep only atoms whose molecule-ID (2nd column of the Atoms section)
    equals `molecule_id`, plus any bonds/angles/dihedrals/impropers whose
    atoms are all kept.

    Unlike `filter_by_distance`, this does NOT renumber atom IDs -- it
    just drops non-matching rows -- so that a subsequent
    `--center-atom`/`filter_by_distance` pass (which expects original,
    pre-merge numbering) still works correctly. Call `renumber_atoms`
    afterwards if no further filtering will be applied.
    """
    atoms = sections.get("Atoms", [])
    keep_ids = {int(row[0]) for row in atoms if row[1] == str(molecule_id)}
    if not keep_ids:
        raise ValueError(f"No atoms found with molecule-ID {molecule_id}.")

    new_sections = {"Atoms": [row for row in atoms if int(row[0]) in keep_ids]}
    new_sections["_known_edge_ids"] = {
        aid for aid in sections.get("_known_edge_ids", set()) if aid in keep_ids
    }
    # Preserve type-label sections unchanged; type numbers themselves
    # aren't touched by this filtering step.
    for label_section in LABEL_SECTIONS.values():
        if label_section in sections:
            new_sections[label_section] = sections[label_section]
    for section, natoms in (
        ("Bonds", 2), ("Angles", 3), ("Dihedrals", 4), ("Impropers", 4),
    ):
        new_rows = []
        for row in sections.get(section, []):
            atom_fields = [int(row[k]) for k in range(2, 2 + natoms)]
            if all(a in keep_ids for a in atom_fields):
                new_row = list(row)
                new_row[0] = str(len(new_rows) + 1)
                new_rows.append(new_row)
        new_sections[section] = new_rows

    new_counts = dict(counts)
    new_counts["atoms"] = len(new_sections["Atoms"])
    for section, label in (
        ("Bonds", "bonds"), ("Angles", "angles"),
        ("Dihedrals", "dihedrals"), ("Impropers", "impropers"),
    ):
        new_counts[label] = len(new_sections[section])

    return new_counts, new_sections


def renumber_atoms(counts, sections):
    """Renumber atom IDs contiguously starting at 1 (in ascending order of
    original ID), along with the atom IDs referenced in bonds/angles/
    dihedrals/impropers, and renumber those sections' own IDs
    contiguously too. Assumes every bond/angle/dihedral/improper row
    already references only atoms present in `sections["Atoms"]`.
    """
    atoms = sections.get("Atoms", [])
    id_map = {
        int(row[0]): new_id
        for new_id, row in enumerate(
            sorted(atoms, key=lambda row: int(row[0])), start=1
        )
    }

    new_atoms = []
    for row in atoms:
        new_row = list(row)
        new_row[0] = str(id_map[int(row[0])])
        new_atoms.append(new_row)

    new_sections = dict(sections)
    new_sections["Atoms"] = new_atoms
    new_sections["_known_edge_ids"] = {
        id_map[aid] for aid in sections.get("_known_edge_ids", set()) if aid in id_map
    }
    for section, natoms in (
        ("Bonds", 2), ("Angles", 3), ("Dihedrals", 4), ("Impropers", 4),
    ):
        new_rows = []
        for row in sections.get(section, []):
            new_row = list(row)
            new_row[0] = str(len(new_rows) + 1)
            for k in range(2, 2 + natoms):
                new_row[k] = str(id_map[int(row[k])])
            new_rows.append(new_row)
        new_sections[section] = new_rows

    return counts, new_sections


def filter_by_distance(counts, sections, center_atom, max_bonds, extra_centers=()):
    """Keep only atoms within `max_bonds` bonds of `center_atom` or of any
    atom in `extra_centers` (inclusive), plus any bonds/angles/
    dihedrals/impropers whose atoms are all kept.

    `extra_centers` (original numbering) lets other atoms of interest
    -- e.g. atoms whose type will change after a reaction, so enough of
    their own neighborhood is kept to correctly determine any new bond/
    angle/dihedral/improper types involving them -- also act as BFS
    roots for inclusion, even if they'd otherwise fall outside
    `max_bonds` hops of `center_atom` alone. They do not become
    additional `center_new_id`s -- only `center_atom` does.

    Atom IDs are renumbered contiguously starting at 1 (in ascending order
    of their original ID), and bond/angle/dihedral/improper IDs are
    renumbered contiguously as well, since LAMMPS molecule templates for
    `fix bond/react` expect a compact fragment.

    Returns (new_counts, new_sections, center_new_id, edge_new_ids, id_map):
      - center_new_id: the renumbered (local, pre-merge) atom ID of
        `center_atom`, for use as a `fix bond/react` InitiatorID.
      - edge_new_ids: sorted list of renumbered (local, pre-merge) atom
        IDs of "edge" atoms, i.e. kept atoms that were originally bonded
        to an atom excluded by the filter, for use as `fix bond/react`
        EdgeIDs.
      - id_map: {original_atom_id: renumbered (local, pre-merge) atom_id}
        for every atom kept by the filter (original numbering -> local
        numbering), e.g. for translating other atom IDs given in the
        file's own original numbering (including any of `extra_centers`).

    Raises ValueError if `sections` carries pre-existing EdgeIDs (from a
    `# Known EdgeIDs:` comment, i.e. a fragment boundary from an earlier
    extraction/reaction step) that sit fewer than `max_bonds` hops from
    `center_atom`: such a boundary means this fragment's own file has no
    record of what lies beyond it, so the neighborhood this filter is
    meant to keep around `center_atom` (for correctly determining any
    new bond/angle/dihedral/improper formed there) may actually be
    incomplete.
    """
    atoms = sections.get("Atoms", [])
    bonds = sections.get("Bonds", [])

    atom_ids = {int(row[0]) for row in atoms}
    all_centers = [center_atom] + list(extra_centers)
    missing = [c for c in all_centers if c not in atom_ids]
    if missing:
        raise ValueError(
            f"Atom(s) {', '.join(str(m) for m in missing)} not found "
            f"among the molecule's {len(atom_ids)} atoms."
        )

    adjacency = {aid: set() for aid in atom_ids}
    for row in bonds:
        a1, a2 = int(row[2]), int(row[3])
        adjacency[a1].add(a2)
        adjacency[a2].add(a1)

    def bfs_within(root):
        """(Atom IDs within `max_bonds` hops of `root` (inclusive),
        {atom_id: hop distance from `root`} for those same atoms)."""
        distances = {root: 0}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            if distances[current] >= max_bonds:
                continue
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return set(distances), distances

    # Union of BFS out to max_bonds hops from center_atom and from every
    # extra_centers atom, so e.g. a retyped atom sitting near the edge of
    # center_atom's own neighborhood still gets its own full max_bonds
    # neighborhood kept.
    keep_ids = set()
    center_distances = None
    for root in all_centers:
        ids, distances = bfs_within(root)
        keep_ids |= ids
        if root == center_atom:
            center_distances = distances

    # Any atom already known (e.g. from a `# Known EdgeIDs:` comment
    # carried over from an earlier extraction/reaction step) to be a
    # fragment-boundary atom represents topology this filter has no way
    # to see past (the real molecule may continue beyond it in ways this
    # fragment's own file simply doesn't record). If such an atom sits
    # closer than `max_bonds` hops from `center_atom`, some of the
    # topology this filter is supposed to keep near `center_atom` (e.g.
    # for a new bond/angle/dihedral/improper formed there) may actually
    # lie beyond that unrecorded boundary -- so refuse to silently build
    # a possibly-incomplete template instead of guessing.
    prior_known_edge_ids = sections.get("_known_edge_ids", set())
    too_close = {
        aid: center_distances[aid]
        for aid in prior_known_edge_ids
        if aid in center_distances and center_distances[aid] < max_bonds
    }
    if too_close:
        details = ", ".join(
            f"atom {aid} ({dist} bond(s) away)"
            for aid, dist in sorted(too_close.items())
        )
        raise ValueError(
            f"pre-existing EdgeID(s) too close to atom {center_atom}: "
            f"{details} -- each is less than --max-bonds ({max_bonds}) "
            f"hops away, so this fragment's own recorded topology may "
            f"be incomplete within the neighborhood this filter is "
            f"supposed to keep around atom {center_atom}. Re-extract "
            f"this fragment with a boundary farther from atom "
            f"{center_atom}, or reduce --max-bonds."
        )

    # Edge atoms: kept atoms that were originally bonded to an atom that
    # got excluded by the filter (i.e. their topology was truncated),
    # plus any atoms already known (e.g. from a `# Known EdgeIDs:`
    # comment carried over from an earlier extraction/reaction step) to
    # sit at the fragment's boundary even though none of their own
    # bonded neighbors happened to be excluded by *this* filter.
    known_edge_ids = prior_known_edge_ids & keep_ids
    edge_ids = known_edge_ids | {
        aid for aid in keep_ids
        if any(neighbor not in keep_ids for neighbor in adjacency[aid])
    }

    # Renumber kept atoms contiguously starting at 1, preserving order.
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(keep_ids), start=1)}

    new_atoms = []
    for row in atoms:
        old_id = int(row[0])
        if old_id in keep_ids:
            new_row = list(row)
            new_row[0] = str(id_map[old_id])
            new_atoms.append(new_row)

    new_sections = {"Atoms": new_atoms}
    # Preserve type-label sections unchanged; type numbers themselves
    # aren't touched by this filtering step.
    for label_section in LABEL_SECTIONS.values():
        if label_section in sections:
            new_sections[label_section] = sections[label_section]
    for section, natoms in (
        ("Bonds", 2), ("Angles", 3), ("Dihedrals", 4), ("Impropers", 4),
    ):
        new_rows = []
        for row in sections.get(section, []):
            atom_fields = [int(row[k]) for k in range(2, 2 + natoms)]
            if all(a in keep_ids for a in atom_fields):
                new_row = list(row)
                new_row[0] = str(len(new_rows) + 1)
                for k, a in zip(range(2, 2 + natoms), atom_fields):
                    new_row[k] = str(id_map[a])
                new_rows.append(new_row)
        new_sections[section] = new_rows

    new_counts = dict(counts)
    new_counts["atoms"] = len(new_atoms)
    for section, label in (
        ("Bonds", "bonds"), ("Angles", "angles"),
        ("Dihedrals", "dihedrals"), ("Impropers", "impropers"),
    ):
        new_counts[label] = len(new_sections[section])

    center_new_id = id_map[center_atom]
    edge_new_ids = sorted(id_map[aid] for aid in edge_ids)

    return new_counts, new_sections, center_new_id, edge_new_ids, id_map


def fragment_canonical_signature(counts, sections, center_new_id, edge_new_ids):
    """Build a canonical signature for a filtered fragment (the same
    shape `filter_by_distance` returns, minus `id_map`), invariant to
    atom renumbering, so two fragments built from different files/
    reactants but that happen to describe the *same* local topology
    once filtered by `--max-bonds` can be recognized as equivalent --
    e.g. a growing product where the ligand furthest from the reacting
    atom got entirely trimmed away, so two different histories produce
    an identical neighborhood around the reacting atom.

    Uses 1-dimensional Weisfeiler-Leman color refinement: every atom
    starts colored by `(its type label, whether it's an EdgeID, whether
    it's center_new_id)`, then is repeatedly recolored by `(its current
    color, the sorted multiset of its bonded neighbors' current colors)`
    until the number of distinct colors stops increasing. Two fragments
    with the same final color for their own `center_new_id`, and the
    same multiset of every atom's final color, describe the same
    topology (this is a reliable, if not mathematically airtight,
    isomorphism test for ordinary organic/coordination fragments; it
    can only fail to distinguish two genuinely different but unusually
    symmetric fragments, never mistake two atoms' roles the other way
    around).

    Colors are never compacted to small per-fragment integers -- each
    atom's color stays the full nested tuple built up from the real,
    shared atom-type-label strings, so two fragments are only ever
    considered to have "the same color" if they actually trace back to
    the same type labels in the same bonding pattern. (An earlier
    implementation reduced each round's colors to integers ranked by
    that fragment's own sorted set of distinct values; since that
    ranking was done independently per fragment, two fragments with
    different but equally-shaped, fully-asymmetric neighborhoods -- e.g.
    two different single-ring reactants where every ring atom is
    topologically distinguishable -- could be assigned the same integers
    purely by coincidence of shape, even though their real atom types
    differed. Keeping the full label tuples avoids that false match.)

    Returns a hashable, comparable tuple:
    `(n_atoms, n_bonds, n_angles, n_dihedrals, n_impropers, center_color,
    canonical_multiset)`, suitable as a dict/set key, or for combining
    two reactants' signatures into an order-independent reaction-pair
    signature (see data2reaction.py's reactant_pair_structural_signature).
    """
    atom_type_label = extract_label_map(sections, "atom")
    type_num_by_atom = {int(row[0]): int(row[2]) for row in sections.get("Atoms", [])}
    label_by_atom = {
        aid: atom_type_label.get(t, str(t)) for aid, t in type_num_by_atom.items()
    }
    edge_id_set = set(edge_new_ids)

    adjacency = {aid: set() for aid in type_num_by_atom}
    for row in sections.get("Bonds", []):
        a1, a2 = int(row[2]), int(row[3])
        adjacency[a1].add(a2)
        adjacency[a2].add(a1)

    color = {
        aid: (label_by_atom[aid], aid in edge_id_set, aid == center_new_id)
        for aid in adjacency
    }

    prev_n_classes = 0
    for _ in range(len(adjacency) + 1):
        color = {
            aid: (color[aid], tuple(sorted(color[n] for n in adjacency[aid])))
            for aid in adjacency
        }
        n_classes = len(set(color.values()))
        if n_classes == prev_n_classes:
            break
        prev_n_classes = n_classes

    return (
        counts.get("atoms", 0), counts.get("bonds", 0), counts.get("angles", 0),
        counts.get("dihedrals", 0), counts.get("impropers", 0),
        color[center_new_id], tuple(sorted(color.values())),
    )


def write_map_file(outpath, n_atoms, initiator_ids, edge_ids, title=None):
    """Write a `fix bond/react` map file.

    `initiator_ids` and `edge_ids` are atom IDs in the final (merged,
    post-filter) molecule template numbering. `n_atoms` is the total atom
    count of that template. Since this script only produces one molecule
    template (not a pre-/post-reaction pair), Equivalences is written as
    the identity mapping, i.e. it assumes the post-reaction template (if
    any) uses the same atom numbering.
    """
    with open(outpath, "w") as f:
        f.write(title or "LAMMPS bond/react map file (generated by data2molecule.py)\n")
        f.write("\n")
        f.write(f"{len(edge_ids)} edgeIDs\n")
        f.write(f"{n_atoms} equivalences\n")
        f.write("\n")

        f.write("InitiatorIDs\n\n")
        for aid in initiator_ids:
            f.write(f"{aid}\n")
        f.write("\n")

        if edge_ids:
            f.write("EdgeIDs\n\n")
            for aid in edge_ids:
                f.write(f"{aid}\n")
            f.write("\n")

        f.write("Equivalences\n\n")
        for aid in range(1, n_atoms + 1):
            f.write(f"{aid}\t{aid}\n")


def write_molecule_file(
    counts, sections, outpath, label_maps=None, title=None, known_edge_ids=None,
    nocharges=False,
):
    atoms = sections.get("Atoms", [])
    bonds = sections.get("Bonds", [])
    angles = sections.get("Angles", [])
    dihedrals = sections.get("Dihedrals", [])
    impropers = sections.get("Impropers", [])

    if not atoms:
        raise ValueError("No 'Atoms' section found in data file.")

    label_maps = label_maps or {}

    def type_str(kind, type_num):
        """Return the type label for type_num if one is defined for this
        kind, otherwise fall back to the plain numeric type string.
        """
        return label_maps.get(kind, {}).get(int(type_num), str(type_num))

    # atom_style full: id mol type q x y z [nx ny nz]
    n_atoms = len(atoms)
    n_bonds = len(bonds)
    n_angles = len(angles)
    n_dihedrals = len(dihedrals)
    n_impropers = len(impropers)

    with open(outpath, "w") as f:
        f.write(title or "LAMMPS molecule template (generated by data2molecule.py)\n")
        f.write("\n")
        if known_edge_ids:
            ids_str = " ".join(str(aid) for aid in sorted(known_edge_ids))
            f.write(
                f"# Known EdgeIDs: {ids_str}\n"
                "# (atoms known to sit at the boundary of this fragment, "
                "e.g. from --map-file's own EdgeID detection or carried "
                "forward from an earlier extraction/reaction step; "
                "redundant with any corresponding map file's EdgeIDs "
                "section, but lets a later data2molecule.py run pick "
                "them up automatically)\n"
            )
            f.write("\n")
        f.write(f"{n_atoms} atoms\n")
        if n_bonds:
            f.write(f"{n_bonds} bonds\n")
        if n_angles:
            f.write(f"{n_angles} angles\n")
        if n_dihedrals:
            f.write(f"{n_dihedrals} dihedrals\n")
        if n_impropers:
            f.write(f"{n_impropers} impropers\n")
        f.write("\n")

        # Coords section: id x y z
        f.write("Coords\n\n")
        for row in atoms:
            atom_id, x, y, z = row[0], row[4], row[5], row[6]
            f.write(f"{atom_id} {x} {y} {z}\n")
        f.write("\n")

        # Types section: id type
        f.write("Types\n\n")
        for row in atoms:
            atom_id, atype = row[0], row[2]
            f.write(f"{atom_id} {type_str('atom', atype)}\n")
        f.write("\n")

        # Charges section: id q (omitted if --nocharges, so LAMMPS
        # ignores charge when matching this template against the system)
        if not nocharges:
            f.write("Charges\n\n")
            for row in atoms:
                atom_id, q = row[0], row[3]
                f.write(f"{atom_id} {q}\n")
            f.write("\n")

        if n_bonds:
            f.write("Bonds\n\n")
            for row in bonds:
                # id type atom1 atom2
                bid, btype, a1, a2 = row[0], row[1], row[2], row[3]
                f.write(f"{bid} {type_str('bond', btype)} {a1} {a2}\n")
            f.write("\n")

        if n_angles:
            f.write("Angles\n\n")
            for row in angles:
                # id type atom1 atom2 atom3
                aid, atype, a1, a2, a3 = row[0], row[1], row[2], row[3], row[4]
                f.write(f"{aid} {type_str('angle', atype)} {a1} {a2} {a3}\n")
            f.write("\n")

        if n_dihedrals:
            f.write("Dihedrals\n\n")
            for row in dihedrals:
                # id type atom1 atom2 atom3 atom4
                did, dtype, a1, a2, a3, a4 = row[:6]
                f.write(f"{did} {type_str('dihedral', dtype)} {a1} {a2} {a3} {a4}\n")
            f.write("\n")

        if n_impropers:
            f.write("Impropers\n\n")
            for row in impropers:
                iid, itype, a1, a2, a3, a4 = row[:6]
                f.write(f"{iid} {type_str('improper', itype)} {a1} {a2} {a3} {a4}\n")
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS data file(s) and/or LAMMPS molecule "
        "template(s) into a single LAMMPS molecule template, optionally "
        "truncated to a fragment centered on a given atom (e.g. for use "
        "with `fix bond/react`)."
    )
    parser.add_argument(
        "data_files", nargs="+",
        help="One or more input files, each either a LAMMPS data file or "
        "a LAMMPS molecule template (e.g. a post-reaction template "
        "produced by react_molecule.py). The format of each file is "
        "auto-detected.",
    )
    parser.add_argument("molecule_file", help="Output molecule template file.")
    parser.add_argument(
        "--center-atom", "--atom", dest="center_atom", type=int, default=None,
        metavar="ID", action="append",
        help="Atom ID, in that file's own (original) numbering, to "
        "center that fragment on. Repeat this flag once per input file, in "
        "the same order as the input files (e.g. `--center-atom 5 "
        "--center-atom 3` for two files). If given, only atoms within "
        "--max-bonds bonds of each file's center atom are kept from that "
        "file, before the fragments are merged. Default: keep the whole "
        "molecule.",
    )
    parser.add_argument(
        "--molecule-id", "--mol-id", dest="molecule_id", type=int, default=None,
        metavar="ID", action="append",
        help="Molecule-ID, matched against the 2nd column (mol) of that "
        "data file's Atoms section, to keep from that file. Repeat this "
        "flag once per input file, in the same order as the input files, "
        "for use with data files (e.g. ZN-IM-data.lammps) containing "
        "multiple distinct molecules. If given, only atoms whose Atoms "
        "mol column matches are kept from that file, before "
        "--center-atom filtering (if any) and before the fragments are "
        "merged. Default: keep every molecule-ID in the file. Molecule "
        "template input files describe a single molecule (mol column is "
        "synthesized as 1 for every atom), so --molecule-id only makes "
        "sense for them if 1 is the value requested.",
    )
    parser.add_argument(
        "--max-bonds", dest="max_bonds", type=int, default=3, metavar="N",
        help="Maximum number of bonds separating an atom from its file's "
        "--center-atom for it to be kept. Applied to every input file. "
        "Only used if --center-atom is given. Default: 3.",
    )
    parser.add_argument(
        "--map-file", dest="map_file", default=None, metavar="PATH",
        help="Also write a `fix bond/react` map file to PATH, containing "
        "InitiatorIDs (the final-numbering IDs of each file's "
        "--center-atom), EdgeIDs (final-numbering IDs of kept atoms that "
        "were bonded to an atom excluded by --max-bonds filtering, "
        "unioned with any atoms already marked via a `# Known EdgeIDs:` "
        "comment in an input molecule template -- e.g. one written by "
        "react_molecule.py's --map-file option -- since such atoms may "
        "already be at the fragment's reactive boundary even though "
        "none of their own neighbors were excluded by this run), and an "
        "identity Equivalences mapping. Requires --center-atom.",
    )
    parser.add_argument(
        "--nocharges", dest="nocharges", action="store_true",
        help="Omit the Charges section from the output molecule "
        "template, so e.g. `fix bond/react` ignores atom charge "
        "entirely when matching this template against the system.",
    )
    args = parser.parse_args()

    if args.data_files[0] == args.molecule_file and len(args.data_files) == 1:
        parser.error("input file and output molecule file must be different")
    if args.max_bonds < 0:
        parser.error("--max-bonds must be >= 0")
    if args.center_atom is not None and len(args.center_atom) != len(args.data_files):
        parser.error(
            f"--center-atom requires exactly one atom ID per input file "
            f"({len(args.data_files)} given), got {len(args.center_atom)}"
        )
    if args.molecule_id is not None and len(args.molecule_id) != len(args.data_files):
        parser.error(
            f"--molecule-id requires exactly one molecule-ID per input file "
            f"({len(args.data_files)} given), got {len(args.molecule_id)}"
        )
    if args.map_file is not None and args.center_atom is None:
        parser.error("--map-file requires --center-atom to be given")

    return args


def main():
    args = parse_args()
    data_paths, mol_path = args.data_files, args.molecule_file

    fragments = [parse_fragment_file(p) for p in data_paths]

    mol_id_args = args.molecule_id or [None] * len(data_paths)
    for path, molecule_id, (_, sections) in zip(data_paths, mol_id_args, fragments):
        if molecule_id is not None:
            continue  # will be narrowed to a single molecule-ID below
        n_mol_ids = len({row[1] for row in sections.get("Atoms", [])})
        if n_mol_ids > 1:
            print(
                f"Warning: {path} contains {n_mol_ids} distinct molecule-IDs; "
                "molecule templates describe a single molecule. Proceeding "
                "using all atoms as one molecule. Use --molecule-id to keep "
                "only one of them."
            )

    if args.molecule_id is not None:
        filtered_fragments = []
        for path, molecule_id, (counts, sections) in zip(
            data_paths, args.molecule_id, fragments
        ):
            f_counts, f_sections = filter_by_molecule_id(counts, sections, molecule_id)
            print(
                f"{path}: filtered to {f_counts['atoms']} atoms with "
                f"molecule-ID {molecule_id}"
            )
            filtered_fragments.append((f_counts, f_sections))
        fragments = filtered_fragments

    initiator_ids = []
    edge_ids = []
    if args.center_atom is not None:
        filtered_fragments = []
        atom_offset = 0
        for path, center_atom, (counts, sections) in zip(
            data_paths, args.center_atom, fragments
        ):
            f_counts, f_sections, center_new_id, f_edge_ids, _ = filter_by_distance(
                counts, sections, center_atom, args.max_bonds
            )
            print(
                f"{path}: filtered to {f_counts['atoms']} atoms within "
                f"{args.max_bonds} bond(s) of atom {center_atom} (original "
                "numbering)"
            )
            initiator_ids.append(center_new_id + atom_offset)
            edge_ids.extend(eid + atom_offset for eid in f_edge_ids)
            atom_offset += f_counts["atoms"]
            filtered_fragments.append((f_counts, f_sections))
        fragments = filtered_fragments
    elif args.molecule_id is not None:
        # No BFS filtering (which renumbers as a side effect) was applied,
        # but molecule-ID filtering may have left gaps in atom numbering,
        # so compact each fragment's IDs before writing/merging.
        fragments = [renumber_atoms(counts, sections) for counts, sections in fragments]

    if len(fragments) == 1:
        counts, sections = fragments[0]
        label_maps = {kind: extract_label_map(sections, kind) for kind in LABEL_SECTIONS}
    else:
        counts, sections, label_maps = merge_fragments(fragments)
        print(f"Merged {len(fragments)} data files: {', '.join(data_paths)}")

    # When --center-atom filtering ran, `edge_ids` already reflects the
    # final numbering (BFS-detected edges unioned with any known edges
    # carried over from the input files). Otherwise, fall back to
    # whatever known edge IDs survived unfiltered in `sections`, so they
    # still get carried forward into the output template's comments.
    final_edge_ids = edge_ids if args.center_atom is not None else sorted(
        sections.get("_known_edge_ids", set())
    )

    write_molecule_file(
        counts, sections, mol_path, label_maps=label_maps, known_edge_ids=final_edge_ids,
        nocharges=args.nocharges,
    )
    print(f"Wrote molecule template to {mol_path}")

    if args.map_file is not None:
        write_map_file(args.map_file, counts["atoms"], initiator_ids, edge_ids)
        print(f"Wrote bond/react map file to {args.map_file}")


if __name__ == "__main__":
    main()
