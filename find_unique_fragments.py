#!/usr/bin/env python3
"""
find_unique_fragments.py

Scan a LAMMPS data file for every atom of a user-specified "initiator"
atom types (given as either a numeric atom type or an Atom Type Labels
label), and determine how many structurally distinct local fragments
those initiator atoms sit at the center of.

For each initiator atom, the fragment is the set of atoms within
--max-bonds bonds of it (default: 3), exactly as `data2molecule.py`'s
--center-atom/--max-bonds filtering would extract it. Two initiator
atoms' fragments are considered the "same" fragment if they are
isomorphic once atom-type labels and bonding topology are taken into
account (i.e. the same test `data2reaction.py` already uses to
recognize equivalent reactant fragments) -- so it is unnecessary (and
unneeded, since they are fully determined by atom types and bonds) to
separately compare angles/dihedrals/impropers.

This reuses `data2molecule.py`'s own `parse_data_file`,
`filter_by_distance`, and `fragment_canonical_signature` functions, so
its notion of a "fragment" and of "unique" is identical to the one
`data2molecule.py`/`data2reaction.py` already rely on elsewhere.

Usage:
    python3 find_unique_fragments.py data.lammps --atom-type fe,nd,nc
    python3 find_unique_fragments.py data.lammps --atom-type 9 --max-bonds 2
    python3 find_unique_fragments.py data.lammps --atom-type nd --list-members
"""

import sys
import argparse

import data2molecule as d2m


def resolve_atom_type(type_spec, atom_type_label_map, n_atom_types):
    """Resolve `type_spec` (a plain integer atom type number, or a type
    label present in the data file's 'Atom Type Labels' section) to
    (type_num, label). Exits with an error message if `type_spec` isn't
    valid for this data file.
    """
    rv = []
    for ts in type_spec:
        if ts.isdigit():
            type_num = int(ts)
            if type_num < 1 or type_num > n_atom_types:
                sys.exit(
                    f"--atom-type '{ts}' is not a valid atom type "
                    f"(the data file declares {n_atom_types} atom type(s))"
                )
            rv.append(type_num)
            continue

        label_to_num = {label: num for num, label in atom_type_label_map.items()}
        if ts not in label_to_num:
            sys.exit(
                f"--atom-type label '{ts}' was not found in the data "
                f"file's 'Atom Type Labels' section"
            )
        rv.append(label_to_num[ts])
    return rv
    

def find_initiator_atoms(sections, type_nums):
    """Return the sorted list of atom IDs (original numbering) whose
    Atoms-section type number is one of `type_nums`."""
    return sorted(
        int(row[0]) for row in sections.get("Atoms", []) if int(row[2]) in type_nums
    )


def fragment_signature_for_initiator(counts, sections, initiator_id, max_bonds):
    """Extract the fragment centered on `initiator_id` (within
    `max_bonds` bonds) and return its canonical (renumbering-invariant)
    signature, via data2molecule.py's own filter_by_distance /
    fragment_canonical_signature -- the same equivalence test
    data2reaction.py uses to recognize equivalent reactant fragments.
    """
    frag_counts, frag_sections, center_new_id, edge_new_ids, _id_map = (
        d2m.filter_by_distance(counts, sections, initiator_id, max_bonds)
    )
    return d2m.fragment_canonical_signature(
        frag_counts, frag_sections, center_new_id, edge_new_ids
    )


def parse_args():
    def list_of_types(arg):
        return list(arg.split(","))

    parser = argparse.ArgumentParser(
        description="Find unique local fragments (within --max-bonds "
        "bonds) around every atom of a given initiator atom type in a "
        "LAMMPS data file."
    )
    parser.add_argument("data_file", help="Input LAMMPS data file.")
    parser.add_argument(
        "--atom-type", "--type", dest="atom_types", type=list_of_types, required=True, metavar="TYPES",
        help="One or more initiator atom type to search for: either a numeric atom "
        "type or typelabel.  Multiple atom types can be listed with commas."
    )
    parser.add_argument(
        "--max-bonds", dest="max_bonds", type=int, default=3, metavar="N",
        help="Maximum number of bonds separating an atom from its "
        "initiator atom for it to be included in that initiator's "
        "fragment. Default: 3.",
    )
    parser.add_argument(
        "--list-members", dest="list_members", action="store_true",
        help="Also list every initiator atom ID sharing each unique "
        "fragment, not just one representative InitiatorID per unique "
        "fragment.",
    )
    parser.add_argument(
        "--reactions", dest="print_reactions", action="store_true",
        help="Print output in format of reactions file."
    )
    args = parser.parse_args()

    if args.max_bonds < 0:
        parser.error("--max-bonds must be >= 0")

    return args


def main():
    args = parse_args()
    reactionLines = find_unique_fragments(args.data_file, args.atom_types,
                          args.max_bonds, args.list_members)
    if args.print_reactions:
        with open("reactions", "w") as f:
            f.write("Reactants\n\n")
            for l in reactionLines:
                f.write(l+"\n")
            f.write("\n")

def find_unique_fragments(data_file, atom_types, max_bonds, list_members):
    counts, sections, extra_lines = d2m.parse_data_file(data_file)

    atom_type_label_map = d2m.extract_label_map(sections, "atom")
    n_atom_types = counts.get("atom types", 0)
    type_nums = resolve_atom_type(
        atom_types, atom_type_label_map, n_atom_types
    )

    initiator_ids = find_initiator_atoms(sections, type_nums)
    if not initiator_ids:
        sys.exit(
            f"No atoms of type '{atom_types}' (type "+", ".join(str(n) for n in type_nums)+") "
            f"found in {data_file}."
        )

    print(
        f"Found {len(initiator_ids)} initiator atom(s)."
    )

    # Map each unique fragment signature -> list of initiator atom IDs
    # (in first-seen order) whose fragment matches it.
    groups = {}
    order = []
    atom_types = {row[0]: row[2] for row in sections["Atoms"]}

    for initiator_id in initiator_ids:
        signature = fragment_signature_for_initiator(
            counts, sections, initiator_id, max_bonds
        )
        if signature not in groups:
            groups[signature] = []
            order.append(signature)

        groups[signature].append(initiator_id)

    print(
        f"Found {len(order)} unique fragment(s) (--max-bonds="
        f"{max_bonds})."
    )

    reactionLines = []
    for i, signature in enumerate(order, start=1):
        members = groups[signature]
        representative = str(members[0])
        t = int(atom_types[representative])
        if t in atom_type_label_map:
            t = atom_type_label_map[t]
        reactionLines.append(chr(64+i)+" "+data_file+" "+representative+" # "+str(len(members))+" "+t)

        print(
            f"  Unique fragment #{i}: InitiatorID {representative} "
            f"({len(members)} initiator atom(s) share this fragment)"
        )
        if list_members:
            member_list = ", ".join(str(m) for m in members)
            print(f"    Members: {member_list}")

    return reactionLines
    

if __name__ == "__main__":
    main()
