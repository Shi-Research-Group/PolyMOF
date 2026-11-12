#!/usr/bin/env python3
"""
react_molecule.py

Given a LAMMPS molecule template (as produced by data2molecule.py) that
describes a "pre-reaction" state -- e.g. a metal ion and a ligand that are
not yet bonded to one another -- form a new bond between two of its atoms
and write out the resulting "post-reaction" molecule template, for use
with `fix bond/react`.

Forming the new bond can create brand new angles, dihedrals, and (for
atoms that become 3-coordinate) impropers that didn't exist in the
pre-reaction template. Dihedral types from the main params file can
often stand in for what is physically an improper/out-of-plane term
(e.g. `harmonic` with d=-1, n=2), but that mapping isn't reliable in
general, so new improper topologies are instead matched against genuine
`improper` type entries (typically found in the extra params file) --
never against dihedral types. This script works out every new bond/
angle/dihedral/improper that involves the new bond, looks up its
force-field type in the supplied LAMMPS parameter file(s), and:
  - includes it in the post-reaction template if a matching type is found
    (in either the main params file or the extra params file), or
  - prints a warning and omits it if no matching type is found anywhere.

Existing bonds/angles/dihedrals/impropers from the pre-reaction template
-- i.e. every pair/triplet/quadruplet of atoms whose bonding doesn't
change -- are always carried over unchanged.

Usage:
    python3 react_molecule.py --bond 1 2 \
        --params ZN-IM-params.lammps --extra-params ZN-IM-extra.params \
        ZN-IM-pre.molecule_template ZN-IM-post.molecule_template

    # Override the auto-derived (from the two atoms' Types) new bond type
    # name instead of deriving it from the atom types:
    python3 react_molecule.py --bond 1 2 --bond-type nc-zn \
        --params ZN-IM-params.lammps --extra-params ZN-IM-extra.params \
        ZN-IM-pre.molecule_template ZN-IM-post.molecule_template

    # Also write a merged parameter file: every line of
    # ZN-IM-params.lammps, plus new bond_coeff/angle_coeff/dihedral_coeff
    # lines (with freshly assigned IDs) for any new type that was only
    # found via --extra-params:
    python3 react_molecule.py --bond 1 2 \
        --params ZN-IM-params.lammps --extra-params ZN-IM-extra.params \
        --merged-params ZN-IM-merged-params.lammps \
        ZN-IM-pre.molecule_template ZN-IM-post.molecule_template

    # Also update a bulk LAMMPS data file's Type Labels sections (and
    # header type counts) so its numbering matches the merged params
    # file and post-reaction template:
    python3 react_molecule.py --bond 1 2 \
        --params ZN-IM-params.lammps --extra-params ZN-IM-extra.params \
        --merged-params ZN-IM-merged-params.lammps \
        --data-file ZN-IM-data.lammps --new-data-file ZN-IM-data-new.lammps \
        ZN-IM-pre.molecule_template ZN-IM-post.molecule_template
"""

import sys
import re
import argparse
from collections import defaultdict

SECTION_NAMES = (
    "Coords", "Types", "Charges", "Velocities",
    "Bonds", "Angles", "Dihedrals", "Impropers",
)

COUNT_RE = re.compile(
    r"^\s*(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)\s*$"
)


def clean_line(line):
    return line.split("#", 1)[0].strip()


KNOWN_EDGE_IDS_RE = re.compile(r"^\s*#\s*Known EdgeIDs:\s*(.*)$")


def read_known_edge_ids(path):
    """Scan a molecule template file for a `# Known EdgeIDs: ...` comment
    (as written by this function's counterpart below) and return the set
    of atom IDs it lists, or an empty set if no such comment is present.
    """
    edge_ids = set()
    with open(path) as f:
        for line in f:
            m = KNOWN_EDGE_IDS_RE.match(line)
            if m:
                edge_ids.update(int(tok) for tok in m.group(1).split())
    return edge_ids


def parse_map_file_edge_ids(path):
    """Parse a `fix bond/react` map file (as written by data2molecule.py)
    and return just its EdgeIDs as a set of ints.
    """
    with open(path) as f:
        lines = [clean_line(line) for line in f]
    lines = [l for l in lines if l]

    edge_ids = set()
    i, n = 0, len(lines)
    while i < n:
        if lines[i] == "EdgeIDs":
            i += 1
            while i < n and lines[i] not in (
                "InitiatorIDs", "EdgeIDs", "Equivalences"
            ):
                edge_ids.add(int(lines[i].split()[0]))
                i += 1
        else:
            i += 1
    return edge_ids


def parse_molecule_template(path):
    """Parse a LAMMPS molecule template file (as written by
    data2molecule.py) into a dict of Python data structures.
    """
    lines = []
    title = None
    with open(path) as f:
        for raw in f:
            if title is None and raw.strip():
                title = raw.rstrip("\n")
                continue
            l = clean_line(raw)
            if l:
                lines.append(l)

    counts = {}
    sections = defaultdict(list)
    i, n = 0, len(lines)

    # Header: counts, until the first section keyword.
    while i < n and lines[i] not in SECTION_NAMES:
        m = COUNT_RE.match(lines[i])
        if m:
            counts[m.group(2)] = int(m.group(1))
        i += 1

    while i < n:
        header = lines[i]
        if header not in SECTION_NAMES:
            i += 1
            continue
        i += 1
        rows = []
        while i < n and lines[i] not in SECTION_NAMES:
            rows.append(lines[i].split())
            i += 1
        sections[header] = rows

    coords = {int(r[0]): (r[1], r[2], r[3]) for r in sections.get("Coords", [])}
    types = {int(r[0]): r[1] for r in sections.get("Types", [])}
    charges = {int(r[0]): r[1] for r in sections.get("Charges", [])}

    def parse_topo(name, natoms):
        rows = []
        for r in sections.get(name, []):
            # id type atom1 [atom2 [atom3 [atom4]]]
            atoms = [int(x) for x in r[2:2 + natoms]]
            rows.append({"type": r[1], "atoms": atoms})
        return rows

    bonds = parse_topo("Bonds", 2)
    angles = parse_topo("Angles", 3)
    dihedrals = parse_topo("Dihedrals", 4)
    impropers = parse_topo("Impropers", 4)

    return {
        "title": title,
        "counts": counts,
        "coords": coords,
        "types": types,
        "charges": charges,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "impropers": impropers,
    }


LABEL_LINE_RE = re.compile(
    r"^(bond|angle|dihedral|improper)(_coeff)?\s+(\S+)\s+([^#]*?)\s*(?:#\s*(.*))?$"
)


def parse_params_file(path):
    """Parse either a LAMMPS parameter file (lines like
    `bond_coeff 9 harmonic 165.8 1.88 # nc-zn (...)`) or an "extra
    params" file (lines like `bond nc-zn 165.8 1.88 # (...)`) and return
    a list of entries:
        [{"kind": "bond", "types": ("nc", "zn"), "label": "nc-zn",
          "coeff_body": "165.8 1.88", "orig_id": 9 or None,
          "source": path}, ...]

    `coeff_body` is the style + numeric coefficients exactly as they
    appear on the line (kept as opaque text), so this works whether the
    style in question (e.g. bond_style) is a single style or `hybrid`
    with a per-line style keyword -- we never need to know which.
    """
    entries = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = LABEL_LINE_RE.match(line)
            if not m:
                continue
            kind, is_coeff_style, second_field, coeff_body, comment = m.groups()
            if is_coeff_style:
                # "bond_coeff <id> [style] <val ...> # <label> (...)"
                if not comment:
                    continue
                label = comment.split()[0]
                orig_id = int(second_field) if second_field.isdigit() else None
            else:
                # "bond <label> [style] <val ...> # (...)"
                label = second_field
                orig_id = None
            types_tuple = tuple(label.split("-"))
            entries.append({
                "kind": kind,
                "types": types_tuple,
                "label": label,
                "coeff_body": coeff_body.strip(),
                "orig_id": orig_id,
                "source": path,
            })
    return entries


def find_entry(entries, kind, type_args):
    """Search `entries` (as returned by parse_params_file, possibly from
    multiple files concatenated) for one whose kind matches and whose
    types match `type_args` in a physically-equivalent order:
      - bond (A, B): also matches (B, A)
      - angle (A, B, C): B is the vertex, so (C, B, A) also matches
      - dihedral (A, B, C, D): the reverse chain (D, C, B, A) also matches
    Returns the first matching entry dict, or None.
    """
    reversed_args = tuple(reversed(type_args))
    for entry in entries:
        if entry["kind"] != kind:
            continue
        if entry["types"] == type_args or entry["types"] == reversed_args:
            return entry
    return None


def find_improper_entries(entries, central_type, wing_types):
    """Return every `improper` entry whose first (central) type matches
    `central_type` and whose remaining 3 types are the same multiset as
    `wing_types`, deduplicated by exact type ordering. Force fields such
    as UFF define up to 3 separate improper terms (same 4 atoms,
    different orderings) for one trigonal center, so this can return
    more than one match -- all of them should be added to the output.
    """
    wing_multiset = sorted(wing_types)
    matches = []
    seen = set()
    for entry in entries:
        types = entry["types"]
        if entry["kind"] != "improper" or len(types) != 4 or types[0] != central_type:
            continue
        if sorted(types[1:]) == wing_multiset and types not in seen:
            seen.add(types)
            matches.append(entry)
    return matches


def build_adjacency(bonds):
    adjacency = defaultdict(list)
    for bond in bonds:
        a1, a2 = bond["atoms"]
        adjacency[a1].append(a2)
        adjacency[a2].append(a1)
    return adjacency


def new_angles_for_bond(adjacency, a, b):
    """(u, a, b) for every existing neighbor u of a (other than b), and
    (a, b, w) for every existing neighbor w of b (other than a) -- i.e.
    every angle that has the brand-new bond a-b as one of its two bonds.
    """
    triples = []
    for u in adjacency[a]:
        if u != b:
            triples.append((u, a, b))
    for w in adjacency[b]:
        if w != a:
            triples.append((a, b, w))
    return triples


def new_dihedrals_for_bond(adjacency, a, b):
    """Every 4-atom path (3 consecutive bonds) that uses the brand-new
    bond a-b as one of its 3 bonds -- since a and b were in disconnected
    fragments before, every such path is new.
    """
    quads = []
    # new bond as the middle bond: u-a-b-w
    for u in adjacency[a]:
        if u == b:
            continue
        for w in adjacency[b]:
            if w == a or w == u:
                continue
            quads.append((u, a, b, w))
    # new bond as the last bond, path extends 2 hops back from a: v-u-a-b
    for u in adjacency[a]:
        if u == b:
            continue
        for v in adjacency[u]:
            if v == a or v == b:
                continue
            quads.append((v, u, a, b))
    # new bond as the first bond, path extends 2 hops out from b: a-b-w-x
    for w in adjacency[b]:
        if w == a:
            continue
        for x in adjacency[w]:
            if x == b or x == a:
                continue
            quads.append((a, b, w, x))
    return quads


def new_improper_centers_for_bond(adjacency, a, b):
    """Return each endpoint of the new bond a-b that ends up with exactly
    3 neighbors (a newly-formed trigonal/planar center), as
    (center, [wing_atom1, wing_atom2, wing_atom3]).
    """
    centers = []
    for center in (a, b):
        neighbors = list(adjacency[center])
        if len(neighbors) == 3:
            centers.append((center, neighbors))
    return centers


def write_molecule_template(
    template, outpath, title=None, known_edge_ids=None, nocharges=False
):
    coords, types, charges = template["coords"], template["types"], template["charges"]
    bonds, angles = template["bonds"], template["angles"]
    dihedrals, impropers = template["dihedrals"], template["impropers"]

    atom_ids = sorted(coords)
    n_atoms = len(atom_ids)
    n_bonds, n_angles = len(bonds), len(angles)
    n_dihedrals, n_impropers = len(dihedrals), len(impropers)

    with open(outpath, "w") as f:
        f.write(title or "LAMMPS molecule template (generated by react_molecule.py)\n")
        f.write("\n")
        if known_edge_ids:
            ids_str = " ".join(str(aid) for aid in sorted(known_edge_ids))
            f.write(
                f"# Known EdgeIDs: {ids_str}\n"
                "# (atoms known, e.g. from an earlier data2molecule.py "
                "--map-file extraction, to sit at the boundary of this "
                "fragment; carried forward for a later data2molecule.py "
                "run to pick up automatically, redundant with any "
                "corresponding map file's EdgeIDs section)\n"
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

        f.write("Coords\n\n")
        for aid in atom_ids:
            x, y, z = coords[aid]
            f.write(f"{aid} {x} {y} {z}\n")
        f.write("\n")

        f.write("Types\n\n")
        for aid in atom_ids:
            f.write(f"{aid} {types[aid]}\n")
        f.write("\n")

        if not nocharges:
            f.write("Charges\n\n")
            for aid in atom_ids:
                f.write(f"{aid} {charges[aid]}\n")
            f.write("\n")

        if n_bonds:
            f.write("Bonds\n\n")
            for i, bond in enumerate(bonds, start=1):
                a1, a2 = bond["atoms"]
                f.write(f"{i} {bond['type']} {a1} {a2}\n")
            f.write("\n")

        if n_angles:
            f.write("Angles\n\n")
            for i, angle in enumerate(angles, start=1):
                a1, a2, a3 = angle["atoms"]
                f.write(f"{i} {angle['type']} {a1} {a2} {a3}\n")
            f.write("\n")

        if n_dihedrals:
            f.write("Dihedrals\n\n")
            for i, dih in enumerate(dihedrals, start=1):
                a1, a2, a3, a4 = dih["atoms"]
                f.write(f"{i} {dih['type']} {a1} {a2} {a3} {a4}\n")
            f.write("\n")

        if n_impropers:
            f.write("Impropers\n\n")
            for i, imp in enumerate(impropers, start=1):
                a1, a2, a3, a4 = imp["atoms"]
                f.write(f"{i} {imp['type']} {a1} {a2} {a3} {a4}\n")
            f.write("\n")


def next_available_id(entries, kind):
    ids = [e["orig_id"] for e in entries if e["kind"] == kind and e["orig_id"] is not None]
    return (max(ids) + 1) if ids else 1


def assign_new_ids(new_entries_by_kind, main_entries):
    """Assign a fresh, sequential numeric ID (per kind) to each entry in
    `new_entries_by_kind`, continuing on from the highest ID already used
    for that kind in `main_entries` (parsed from the --params file(s)).
    Returns {kind: [(new_id, entry), ...]}. This is the single source of
    truth for new type numbering, used both for the merged params file
    and for the updated LAMMPS data file's Type Labels, so the two stay
    in sync.
    """
    assigned = {}
    for kind, new_entries in new_entries_by_kind.items():
        next_id = next_available_id(main_entries, kind)
        rows = []
        for entry in new_entries:
            rows.append((next_id, entry))
            next_id += 1
        assigned[kind] = rows
    return assigned


def write_merged_params(params_paths, assigned, outpath):
    """Write `outpath` containing every line of the file(s) in
    `params_paths` verbatim, followed by one new `<kind>_coeff` line per
    (new_id, entry) in `assigned` (see `assign_new_ids`), using the same
    "id ... # label" convention as the rest of the file (the
    atom-type-list label is kept only as a trailing comment).
    """
    with open(outpath, "w") as out:
        for path in params_paths:
            with open(path) as f:
                out.write(f.read())

        for kind in ("bond", "angle", "dihedral", "improper"):
            rows = assigned.get(kind, [])
            if not rows:
                continue
            out.write(f"\n# Added by react_molecule.py for new {kind}(s)\n")
            for new_id, entry in rows:
                out.write(
                    f"{kind}_coeff {new_id} {entry['coeff_body']} # {entry['label']}\n"
                )


TYPE_LABEL_HEADERS = {
    "bond": "Bond Type Labels",
    "angle": "Angle Type Labels",
    "dihedral": "Dihedral Type Labels",
    "improper": "Improper Type Labels",
}

TYPE_COUNT_WORDS = {
    "bond": "bond types",
    "angle": "angle types",
    "dihedral": "dihedral types",
    "improper": "improper types",
}

DATA_COUNT_RE = re.compile(
    r"^(\d+)(\s+)(atom types|bond types|angle types|dihedral types|improper types)(.*)$"
)


def find_type_labels_section(lines, header_name):
    """Return (header_idx, rows_start, rows_end) for a `<Kind> Type
    Labels` section, i.e. `lines[rows_start:rows_end]` are its (non-blank)
    rows and `rows_end` is the index of the blank line (or EOF) that ends
    it. Returns None if the section isn't present in `lines`.
    """
    for i, line in enumerate(lines):
        if line.strip() == header_name:
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            rows_start = j
            k = rows_start
            while k < len(lines) and lines[k].strip() != "":
                k += 1
            return i, rows_start, k
    return None


def write_updated_data_file(datapath, outpath, assigned):
    """Write `outpath` as a copy of the LAMMPS data file at `datapath`,
    with the Bond/Angle/Dihedral/Improper Type Labels sections (and the
    matching `N <kind> types` header counts) extended to include the new
    types in `assigned` (see `assign_new_ids`), using the exact same IDs
    used in the merged params file. The bulk of the file (Masses, Atoms,
    Bonds, Angles, Dihedrals, ...) is left untouched -- new bonded
    *instances* aren't added here, only the new *types* that `fix
    bond/react` will need once it forms them at runtime.
    """
    with open(datapath) as f:
        lines = f.readlines()

    # Find existing "N <kind> types" header count lines.
    count_line_idx = {}
    for i, line in enumerate(lines):
        m = DATA_COUNT_RE.match(line.rstrip("\n"))
        if m:
            for kind, word in TYPE_COUNT_WORDS.items():
                if m.group(3) == word:
                    count_line_idx[kind] = i

    for kind in ("bond", "angle", "dihedral", "improper"):
        rows = assigned.get(kind, [])
        if not rows:
            continue

        # Update (or insert) the "N <kind> types" header count line.
        if kind in count_line_idx:
            i = count_line_idx[kind]
            m = DATA_COUNT_RE.match(lines[i].rstrip("\n"))
            new_count = int(m.group(1)) + len(rows)
            lines[i] = f"{new_count}{m.group(2)}{m.group(3)}{m.group(4)}\n"
        else:
            # No such count line exists yet (e.g. a system with no
            # impropers at all before this reaction) -- insert one right
            # after the last known type-count line, in the usual
            # atom/bond/angle/dihedral/improper order.
            after = None
            for prior_kind in ("dihedral", "angle", "bond"):
                if prior_kind in count_line_idx:
                    after = count_line_idx[prior_kind]
                    break
            if after is None:
                after = next(
                    (i for i, l in enumerate(lines) if l.rstrip("\n").endswith("atom types")),
                    0,
                )
            new_line = f"{len(rows)} {TYPE_COUNT_WORDS[kind]} \n"
            lines.insert(after + 1, new_line)
            # Shift every later recorded index down by 1 line.
            count_line_idx = {k: (v + 1 if v > after else v) for k, v in count_line_idx.items()}
            count_line_idx[kind] = after + 1

        # Update (or create) the "<Kind> Type Labels" section itself.
        header_name = TYPE_LABEL_HEADERS[kind]
        new_rows = [f"{new_id} {entry['label']}\n" for new_id, entry in rows]
        found = find_type_labels_section(lines, header_name)
        if found:
            _, _, rows_end = found
            lines[rows_end:rows_end] = new_rows
        else:
            # No such section exists yet -- create it just before the
            # "Masses" section (Type Labels sections always precede
            # Masses/Atoms/etc. in a LAMMPS data file).
            masses_idx = next(i for i, l in enumerate(lines) if l.strip() == "Masses")
            block = [f"{header_name}\n", "\n"] + new_rows + ["\n"]
            lines[masses_idx:masses_idx] = block

    with open(outpath, "w") as f:
        f.writelines(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Form a new bond in a pre-reaction LAMMPS molecule "
        "template and write out the resulting post-reaction template, "
        "including any new angles/dihedrals/impropers created by that "
        "bond that have matching parameters."
    )
    parser.add_argument("pre_template", help="Pre-reaction molecule template file.")
    parser.add_argument("post_template", help="Output post-reaction molecule template file.")
    parser.add_argument(
        "--bond", dest="bond", type=int, nargs=2, required=True,
        metavar=("ATOM1", "ATOM2"),
        help="The two atom IDs (in the pre-reaction template's own "
        "numbering) to form a new bond between.",
    )
    parser.add_argument(
        "--bond-type", dest="bond_type", default=None, metavar="NAME",
        help="Type name (e.g. `nc-zn`) for the new bond. Default: derive "
        "it from the two atoms' Types-section labels, e.g. `nc-zn` for "
        "atoms typed `nc` and `zn`.",
    )
    parser.add_argument(
        "--params", dest="params", action="append", default=[], metavar="PATH",
        help="A LAMMPS parameter file (e.g. ZN-IM-params.lammps) to search "
        "for bond/angle/dihedral/improper type parameters. Repeatable.",
    )
    parser.add_argument(
        "--extra-params", dest="extra_params", action="append", default=[],
        metavar="PATH",
        help="An extra parameter file (e.g. ZN-IM-extra.params) to also "
        "search for bond/angle/dihedral/improper type parameters. "
        "Repeatable.",
    )
    parser.add_argument(
        "--merged-params", dest="merged_params", default=None, metavar="PATH",
        help="Also write a merged LAMMPS parameter file to PATH, "
        "containing every line of the --params file(s) verbatim plus new "
        "bond_coeff/angle_coeff/dihedral_coeff/improper_coeff lines (with "
        "freshly assigned IDs) for any new bond/angle/dihedral/improper "
        "used in the post-reaction template that wasn't already in "
        "--params (e.g. because it was only found via --extra-params). "
        "Requires --params.",
    )
    parser.add_argument(
        "--map-file", dest="map_file", default=None, metavar="PATH",
        help="An existing `fix bond/react` map file for the pre-reaction "
        "template (e.g. one written by an earlier data2molecule.py "
        "--map-file run), used only to read its EdgeIDs. Those EdgeIDs, "
        "along with any already recorded in a `# Known EdgeIDs:` comment "
        "in --pre-template itself, are written back out as a `# Known "
        "EdgeIDs:` comment in --post-template, so a later data2molecule.py "
        "run on the post-reaction template can recognize them as edge "
        "atoms even if their topology wasn't truncated by that run's own "
        "--max-bonds filtering.",
    )
    parser.add_argument(
        "--data-file", dest="data_file", default=None, metavar="PATH",
        help="An existing LAMMPS data file (e.g. ZN-IM-data.lammps) whose "
        "Bond/Angle/Dihedral/Improper Type Labels sections (and matching "
        "`N <kind> types` header counts) should be extended with the new "
        "types needed for this reaction, using the same numbering as the "
        "merged params file. Requires --new-data-file and --params.",
    )
    parser.add_argument(
        "--new-data-file", dest="new_data_file", default=None, metavar="PATH",
        help="Output path for the updated copy of --data-file.",
    )
    args = parser.parse_args()

    if not args.params and not args.extra_params:
        parser.error("at least one of --params / --extra-params is required")
    if args.bond[0] == args.bond[1]:
        parser.error("--bond requires two different atom IDs")
    if args.merged_params is not None and not args.params:
        parser.error("--merged-params requires --params")
    if (args.data_file is None) != (args.new_data_file is None):
        parser.error("--data-file and --new-data-file must be given together")
    if args.data_file is not None and not args.params:
        parser.error("--data-file requires --params")

    return args


def main():
    args = parse_args()
    template = parse_molecule_template(args.pre_template)

    known_edge_ids = read_known_edge_ids(args.pre_template)
    if args.map_file is not None:
        known_edge_ids |= parse_map_file_edge_ids(args.map_file)
    known_edge_ids &= set(template["types"])  # only atoms actually in the template

    a, b = args.bond
    for atom in (a, b):
        if atom not in template["types"]:
            sys.exit(f"Error: atom {atom} not found in {args.pre_template}")
    existing_bond_pairs = {tuple(sorted(bond["atoms"])) for bond in template["bonds"]}
    if tuple(sorted((a, b))) in existing_bond_pairs:
        sys.exit(f"Error: atoms {a} and {b} are already bonded in {args.pre_template}")

    main_entries = [e for p in args.params for e in parse_params_file(p)]
    extra_entries = [e for p in args.extra_params for e in parse_params_file(p)]
    all_entries = main_entries + extra_entries

    # Entries newly needed in the merged params file, i.e. matched via
    # all_entries but not already present in main_entries (--params).
    new_entries_by_kind = {"bond": [], "angle": [], "dihedral": [], "improper": []}
    added_keys = set()

    def use_entry(kind, type_args):
        entry = find_entry(all_entries, kind, type_args)
        if entry is None:
            return None
        if find_entry(main_entries, kind, type_args) is None:
            key = (kind, entry["types"])
            if key not in added_keys:
                added_keys.add(key)
                new_entries_by_kind[kind].append(entry)
        return entry["label"]

    main_improper_types = {e["types"] for e in main_entries if e["kind"] == "improper"}

    def use_improper_entries(central_type, wing_types):
        matches = find_improper_entries(all_entries, central_type, wing_types)
        for entry in matches:
            if entry["types"] not in main_improper_types:
                key = ("improper", entry["types"])
                if key not in added_keys:
                    added_keys.add(key)
                    new_entries_by_kind["improper"].append(entry)
        return matches

    adjacency = build_adjacency(template["bonds"])

    ta, tb = template["types"][a], template["types"][b]
    if args.bond_type:
        # User-supplied name: look it up directly (trying both atom
        # orders) rather than deriving it from the atoms' own Types.
        parts = args.bond_type.split("-", 1)
        t1, t2 = (parts[0], parts[1]) if len(parts) == 2 else (args.bond_type, args.bond_type)
        found_bond_type = use_entry("bond", (t1, t2))
    else:
        found_bond_type = use_entry("bond", (ta, tb))
    if found_bond_type is None:
        sys.exit(
            f"Error: no bond parameters found (in --params or "
            f"--extra-params) for new bond {a}({ta})-{b}({tb})"
        )
    else:
        template["bonds"].append({"type": found_bond_type, "atoms": [a, b]})

    # Update adjacency now that the new bond exists, so the atoms
    # involved in the new bond show up as each other's neighbors for the
    # angle/dihedral/improper searches below.
    adjacency[a].append(b)
    adjacency[b].append(a)

    for u, va, vb in new_angles_for_bond(adjacency, a, b):
        t1, t2, t3 = template["types"][u], template["types"][va], template["types"][vb]
        label = use_entry("angle", (t1, t2, t3))
        if label is None:
            print(
                f"Warning: no angle parameters found (in --params or "
                f"--extra-params) for new angle {u}({t1})-{va}({t2})-"
                f"{vb}({t3}); omitting from output."
            )
            continue
        template["angles"].append({"type": label, "atoms": [u, va, vb]})

    for quad in new_dihedrals_for_bond(adjacency, a, b):
        tt = tuple(template["types"][x] for x in quad)
        label = use_entry("dihedral", tt)
        if label is None:
            chain = "-".join(f"{x}({t})" for x, t in zip(quad, tt))
            print(
                f"Warning: no dihedral parameters found (in --params or "
                f"--extra-params) for new dihedral {chain}; omitting from "
                "output."
            )
            continue
        template["dihedrals"].append({"type": label, "atoms": list(quad)})

    for center, wing_atoms in new_improper_centers_for_bond(adjacency, a, b):
        central_type = template["types"][center]
        wing_types = [template["types"][nb] for nb in wing_atoms]
        matches = use_improper_entries(central_type, wing_types)
        if not matches:
            print(
                f"Warning: no improper parameters found (in --params or "
                f"--extra-params) for a new trigonal center at atom "
                f"{center} ({central_type}) with substituents "
                f"{'-'.join(wing_types)}; omitting."
            )
            continue
        for entry in matches:
            # entry["types"][0] is the central type; entry["types"][1:]
            # dictates the order the 3 wing atoms must appear in.
            remaining = list(wing_atoms)
            ordered_atoms = [center]
            for wing_type in entry["types"][1:]:
                idx = next(
                    k for k, nb in enumerate(remaining)
                    if template["types"][nb] == wing_type
                )
                ordered_atoms.append(remaining.pop(idx))
            template["impropers"].append({"type": entry["label"], "atoms": ordered_atoms})

    write_molecule_template(template, args.post_template, known_edge_ids=known_edge_ids)
    print(f"Wrote post-reaction molecule template to {args.post_template}")

    if args.merged_params is not None or args.data_file is not None:
        assigned = assign_new_ids(new_entries_by_kind, main_entries)

    if args.merged_params is not None:
        write_merged_params(args.params, assigned, args.merged_params)
        n_new = sum(len(v) for v in assigned.values())
        print(f"Wrote merged parameter file to {args.merged_params} ({n_new} new type(s))")

    if args.data_file is not None:
        write_updated_data_file(args.data_file, args.new_data_file, assigned)
        n_new = sum(len(v) for v in assigned.values())
        print(
            f"Wrote updated data file to {args.new_data_file} ({n_new} new "
            "type(s) added to Type Labels sections)"
        )


if __name__ == "__main__":
    main()
