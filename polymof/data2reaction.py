#!/usr/bin/env python3
"""
data2reaction.py

Combine `data2molecule.py` and `react_molecule.py` into a single command:
build a pre-reaction LAMMPS molecule template from two LAMMPS data files /
molecule templates (exactly like `data2molecule.py`), then immediately
form a new bond in it and write out the resulting post-reaction template
(exactly like `react_molecule.py`), all in one process.

The two atoms to bond are *not* given separately (there is no --bond
flag here): they are simply each reactant's initiator atom, after
data2molecule's own filtering/merging has renumbered them.

The reaction (or sequence/tree of reactions) is described by a
--reactions-file, which can express a single reaction, a linear chain
of reactions (e.g. a Zn ion reacting with one imidazole (IM) ligand,
then that reaction's product reacting with a second, fresh copy of the
same IM fragment, and so on), or an arbitrary tree of reactions. See
--reactions-file below for its format.

Each new bond's type is always derived from the two bonding atoms'
Types-section labels, looked up against --params/--extra-params.

This is purely a convenience wrapper: `data2molecule.py` and
`react_molecule.py` are unchanged and remain fully usable on their own
(e.g. to build/inspect a pre-reaction template by hand, or to react an
existing/hand-edited pre-template file, independent of this script).

Usage:
    python3 data2reaction.py \
        --reactions-file ZN-IM-reactions.txt --max-bonds 2 \
        --params ZN-IM-params.lammps --extra-params ZN-IM-extra.params \
        --merged-params ZN-IM-merged-params.lammps \
        --data-file ZN-IM-data.lammps --new-data-file ZN-IM-data-new.lammps \
        --reaction-name ZN-IM
"""

import sys
import argparse
from collections import deque

import data2molecule as d2m
import react_molecule as rmol


def _find_options_file_path(argv):
    """Scan raw argv (before argparse runs) for --options-file's value,
    without triggering argparse's required-argument checks."""
    for i, tok in enumerate(argv):
        if tok == "--options-file":
            return argv[i + 1] if i + 1 < len(argv) else None
        if tok.startswith("--options-file="):
            return tok.split("=", 1)[1]
    return None


def _read_options_file(parser, path):
    """Parse `path`: each non-blank, non-comment (`#`) line is
    "option-name [value]", where option-name is one of the script's
    long options, without its leading `--`. Returns a list of
    (option_name, value_or_None) tuples, in file order.

    Calls `parser.error()` (which exits) if a line names an unknown
    option or another --options-file.
    """
    long_opts = {
        opt[2:]
        for action in parser._actions
        for opt in action.option_strings
        if opt.startswith("--")
    }
    entries = []
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split(None, 1)
            name = parts[0]
            value = parts[1].strip() if len(parts) > 1 else None
            if name == "options-file":
                parser.error(
                    f"{path}: line {lineno}: an --options-file cannot "
                    f"itself contain an 'options-file' line"
                )
            if name not in long_opts:
                parser.error(
                    f"{path}: line {lineno}: unrecognized option '{name}'"
                )
            entries.append((name, value))
    return entries


def merge_options_file(parser, argv):
    """If --options-file is present in `argv`, read it and merge its
    options in ahead of `argv`'s own options. Any option present in both
    is kept only from `argv` (the command line), with a warning printed.

    Returns the (possibly extended) argv list to actually parse.
    """
    options_file = _find_options_file_path(argv)
    if options_file is None:
        return argv

    cli_opts = {
        tok[2:].split("=", 1)[0] for tok in argv if tok.startswith("--")
    }
    warned = set()
    file_tokens = []
    for name, value in _read_options_file(parser, options_file):
        if name in cli_opts:
            if name not in warned:
                print(
                    f"Warning: --{name} given on the command line "
                    f"overrides its value in {options_file}"
                )
                warned.add(name)
            continue
        file_tokens.append(f"--{name}")
        if value is not None:
            file_tokens.append(value)

    return file_tokens + argv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a pre-reaction LAMMPS molecule template from "
        "two LAMMPS data files/molecule templates (as data2molecule.py "
        "would), then form a new bond between their initiator atoms "
        "and write the resulting post-reaction template (as "
        "react_molecule.py would) -- optionally repeated as a sequence/"
        "tree of reactions -- all in one command, as described by "
        "--reactions-file."
    )

    # --- data2molecule-style arguments ---
    parser.add_argument(
        "--reaction-name", dest="reaction_name", required=True, metavar="NAME",
        help="Base name used to construct the output pre/post-template "
        "and map file names, e.g. `ZN-IM` produces `ZN-IM_1.map`, "
        "`ZN-IM_1-pre.molecule_template`, `ZN-IM_1-post.molecule_template`, "
        "and so on (`ZN-IM_<label>...` per reaction label from "
        "--reactions-file).",
    )
    parser.add_argument(
        "--max-bonds", dest="max_bonds", type=int, default=3, metavar="N",
        help="Maximum number of bonds separating an atom from its "
        "reactant's initiator atom for it to be kept. Applied to every "
        "reactant, every reaction. Default: 3.",
    )
    parser.add_argument(
        "--nocharges", dest="nocharges", action="store_true",
        help="Omit the Charges section from every pre/post-reaction "
        "molecule template written, so e.g. `fix bond/react` ignores "
        "atom charge entirely when matching templates against the "
        "system.",
    )
    parser.add_argument(
        "--skip-equivalent", dest="skip_equivalent", action="store_true",
        help="Only meaningful with --reactions-file. Instead of stopping "
        "with an error when a reaction is equivalent to an earlier one "
        "(same reactants/initiators regardless of side, or -- after "
        "--max-bonds filtering -- the same reacting-atom neighborhood "
        "topology even under different reactant names), print a notice "
        "and skip writing that reaction's templates, then continue with "
        "the rest of the reactions file.",
    )

    # --- reaction-sequence arguments ---
    parser.add_argument(
        "--reactions-file", dest="reactions_file", required=True, metavar="PATH",
        help="A file describing an arbitrary sequence/tree of reactions "
        "(a single reaction, a linear chain, or a tree). Has a "
        "`Reactants` section listing, one per line, a reactant's name, "
        "its input file, and its initiator atom (e.g. "
        "`Zn Zn-data.lammps 5`), and a "
        "`Reactions` section listing, one per line, a reaction's label "
        "and its two reactants (e.g. `1 Zn IM`), each either a Reactants "
        "name (a fresh copy of that reactant) or "
        "`reaction_label:reactant_name` (that reactant's own atom from "
        "that earlier reaction's whole product, re-filtered by "
        "--max-bonds around it just like a fresh copy). An error is "
        "raised if this re-filtering would need to look past a "
        "pre-existing EdgeID (from a `# Known EdgeIDs:` comment) that "
        "sits closer than --max-bonds to that atom. Blank "
        "lines are ignored, and a `#` and everything after it on a line "
        "is treated as a comment and ignored. A Reactants line may also "
        "carry a `posttype atomID:type atomID:type ...` keyword (each "
        "atomID in that reactant's own input-file numbering, each type "
        "a type number or type label resolved against that same file) "
        "giving new atom types to apply only to the post-reaction "
        "molecule template, the first time a fresh copy of that "
        "reactant reacts (see load_reactants_and_reactions for details).",
    )

    # --- react_molecule-style arguments ---
    parser.add_argument(
        "--params", dest="params", action="append", default=[], metavar="PATH",
        help="A LAMMPS parameter file to search for bond/angle/dihedral/"
        "improper type parameters. Repeatable.",
    )
    parser.add_argument(
        "--extra-params", dest="extra_params", action="append", default=[],
        metavar="PATH",
        help="An extra parameter file to also search for bond/angle/"
        "dihedral/improper type parameters. Repeatable.",
    )
    parser.add_argument(
        "--merged-params", dest="merged_params", default=None, metavar="PATH",
        help="Also write a single merged LAMMPS parameter file to PATH, "
        "accumulating every new type discovered across the whole "
        "reaction sequence. Requires --params.",
    )
    parser.add_argument(
        "--data-file", dest="data_file", default=None, metavar="PATH",
        help="An existing LAMMPS data file whose Type Labels sections "
        "(and header type counts) should be extended with the new types "
        "needed across the whole reaction sequence. Requires "
        "--new-data-file and --params.",
    )
    parser.add_argument(
        "--new-data-file", dest="new_data_file", default=None, metavar="PATH",
        help="Output path for the updated copy of --data-file.",
    )
    parser.add_argument(
        "--lammps-reaction-input", dest="lammps_reaction_input", default=None,
        metavar="PATH",
        help="Also write a LAMMPS input-script fragment to PATH: one "
        "`molecule moleculeID molecule_file` line per pre/post-reaction "
        "template file written across the whole reaction sequence, "
        "followed by a `fix ID group-ID bond/react ... react ...` command "
        "with one `react` clause per reaction, using the moleculeIDs and "
        "map files just written. Each moleculeID is "
        "`<reaction-name>_<label>_pre`/`_post` (label is the reaction's "
        "number, or its --reactions-file label; underscores rather than "
        "hyphens are used since LAMMPS molecule template IDs must be "
        "alphanumeric/underscore only); each react-ID is "
        "`<reaction-name>_<label>`. Elements that cannot be filled in "
        "automatically (fix ID, group-ID, react-group-ID, Nevery, Rmin, "
        "Rmax) are left as placeholders for the user to edit.",
    )
    parser.add_argument(
        "--options-file", dest="options_file", default=None, metavar="PATH",
        help="A file of additional command-line options, one per line, "
        "each written as the option's name (without its leading `--`) "
        "followed by its value, e.g. `reaction-name ZN-IM`. Blank lines "
        "are ignored, and a `#` and everything after it on a line is "
        "treated as a comment and ignored. An option given "
        "directly on the command line overrides the same option's value "
        "from this file (a warning is printed when this happens). This "
        "file cannot itself contain an `options-file` line.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print non-essential info about the scripts progress.",
    )

    argv = merge_options_file(parser, sys.argv[1:])
    args = parser.parse_args(argv)

    if args.max_bonds < 0:
        parser.error("--max-bonds must be >= 0")
    if not args.params and not args.extra_params:
        parser.error("at least one of --params / --extra-params is required")
    if args.merged_params is not None and not args.params:
        parser.error("--merged-params requires --params")
    if (args.data_file is None) != (args.new_data_file is None):
        parser.error("--data-file and --new-data-file must be given together")
    if args.data_file is not None and not args.params:
        parser.error("--data-file requires --params")

    return args


def parse_and_filter_fragment(path, initiator_id, max_bonds, extra_centers=()):
    """Parse `path` (data file or molecule template, auto-detected), then
    filter/renumber it down to the atoms within `max_bonds` bonds of
    `initiator_id` (in the file's own, original numbering), or of any
    atom in `extra_centers` (also original numbering; see
    data2molecule.filter_by_distance).

    Returns (counts, sections, center_new_id, edge_new_ids, id_map), same
    as data2molecule.filter_by_distance.
    """
    counts, sections, extra_lines = d2m.parse_fragment_file(path)
    try:
        counts, sections, center_new_id, edge_new_ids, id_map = d2m.filter_by_distance(
            counts, sections, initiator_id, max_bonds, extra_centers=extra_centers
        )
    except ValueError as exc:
        sys.exit(f"{path}: {exc}")
    return counts, sections, center_new_id, edge_new_ids, id_map


def build_and_write_pre_template(fragments_with_centers, pre_path, nocharges=False, verbose=False):
    """fragments_with_centers: list of (counts, sections, center_new_id)
    tuples, one per fragment to merge, in the order they should be
    concatenated (offsets accumulate in this order). A fragment's
    center_new_id may be None if it has no reactive-site atom this round
    (e.g. none needed).

    Writes the merged molecule template to `pre_path` and returns
    (merged_counts, atom_offsets, offset_initiator_ids, final_edge_ids)
    where atom_offsets[i] is the atom-ID offset applied to fragment i,
    offset_initiator_ids[i] is fragment i's center_new_id plus its offset
    (or None), and final_edge_ids is the sorted list of EdgeIDs (in final
    numbering) to use for a `fix bond/react` map file.
    """
    plain_fragments = []
    atom_offsets = []
    offset_initiator_ids = []
    edge_ids = []
    atom_offset = 0
    for counts, sections, center_new_id, *rest in fragments_with_centers:
        edge_new_ids = rest[0] if rest else []
        atom_offsets.append(atom_offset)
        offset_initiator_ids.append(
            None if center_new_id is None else center_new_id + atom_offset
        )
        edge_ids.extend(eid + atom_offset for eid in edge_new_ids)
        plain_fragments.append((counts, sections))
        atom_offset += counts["atoms"]

    if len(plain_fragments) == 1:
        counts, sections = plain_fragments[0]
        label_maps = {
            kind: d2m.extract_label_map(sections, kind) for kind in d2m.LABEL_SECTIONS
        }
    else:
        counts, sections, label_maps = d2m.merge_fragments(plain_fragments)

    final_edge_ids = sorted(set(edge_ids) | sections.get("_known_edge_ids", set()))

    d2m.write_molecule_file(
        counts, sections, pre_path, label_maps=label_maps,
        known_edge_ids=final_edge_ids, nocharges=nocharges,
    )
    if verbose:
        print(f"Wrote pre-reaction molecule template to {pre_path}")

    return counts, atom_offsets, offset_initiator_ids, final_edge_ids


def react_pre_template(
    pre_path, post_path, map_path, bond, all_entries, main_entries,
    added_keys, new_entries_by_kind, missing, charge_adjustments=None,
    type_adjustments=None, nocharges=False, verbose=False
):
    """Equivalent of react_molecule.py's main(), given the new bond's two
    (final, pre-template-numbering) atom IDs. `main_entries`, `all_entries`,
    `added_keys`, and `new_entries_by_kind` are shared/mutated across a
    whole reaction sequence, so a type discovered in an earlier reaction
    is recognized (and not re-added) in later ones.

    `charge_adjustments`, if given, is a list of (atom_id, delta) pairs
    (final, pre-template-numbering): each atom's charge is shifted by
    `delta` in the post-reaction template only (the pre-reaction
    template written by build_and_write_pre_template already carries
    whatever charge that atom should have going into this reaction).
    Ignored if `nocharges` is set (there's no Charges section to shift).

    `type_adjustments`, if given, is a list of (atom_id, new_type_label)
    pairs (final, pre-template-numbering): each atom's type is replaced
    by `new_type_label` for the post-reaction template (the pre-reaction
    template written by build_and_write_pre_template already carries
    each atom's original type, so the reactants' original types are
    always available to react against, and any bonds/angles/dihedrals/
    impropers already present in the pre-reaction template keep
    whatever type label they already had). The new bond formed by this
    reaction, and any new angles/dihedrals/impropers it creates, are
    looked up *after* these type changes are applied, so they reflect
    the post-reaction chemistry (e.g. a new metal-N bond/angle looked up
    using the nitrogen's post-reaction, not pre-reaction, type). A
    warning is printed if an atom's new type is the same as its current
    one.
    """
    template = rmol.parse_molecule_template(pre_path)

    known_edge_ids = rmol.read_known_edge_ids(pre_path)
    known_edge_ids |= rmol.parse_map_file_edge_ids(map_path)
    known_edge_ids &= set(template["types"])

    # Apply posttype type changes to the in-memory template *before*
    # detecting the new bond's type and any new angles/dihedrals/
    # impropers it creates, so those newly-formed interactions are
    # computed using the post-reaction types (e.g. a metal-N bond formed
    # to an atom whose type just flipped from a protonated to an
    # unprotonated nitrogen label). Bonds/angles/dihedrals/impropers
    # that already existed in the pre-reaction template keep whatever
    # type label they already had -- only newly-formed ones look up
    # parameters using the post-reaction types.
    for atom_id, new_label in type_adjustments or []:
        if atom_id not in template["types"]:
            sys.exit(f"Error: atom {atom_id} not found in {pre_path}")
        if template["types"][atom_id] == new_label:
            print(
                f"Warning: posttype gives atom {atom_id} the same type "
                f"({new_label}) it already has in {pre_path}; no change "
                "made."
            )
        template["types"][atom_id] = new_label

    a, b = bond
    for atom in (a, b):
        if atom not in template["types"]:
            sys.exit(f"Error: atom {atom} not found in {pre_path}")
    existing_bond_pairs = {tuple(sorted(bd["atoms"])) for bd in template["bonds"]}
    if tuple(sorted((a, b))) in existing_bond_pairs:
        sys.exit(f"Error: atoms {a} and {b} are already bonded in {pre_path}")

    def use_entry(kind, type_args):
        entry = rmol.find_entry(all_entries, kind, type_args)
        if entry is None:
            return None
        if rmol.find_entry(main_entries, kind, type_args) is None:
            key = (kind, entry["types"])
            if key not in added_keys:
                added_keys.add(key)
                new_entries_by_kind[kind].append(entry)
                main_entries.append(entry)
        return entry["label"]

    def use_improper_entries(central_type, wing_types):
        main_improper_types = {
            e["types"] for e in main_entries if e["kind"] == "improper"
        }
        matches = rmol.find_improper_entries(all_entries, central_type, wing_types)
        for entry in matches:
            if entry["types"] not in main_improper_types:
                key = ("improper", entry["types"])
                if key not in added_keys:
                    added_keys.add(key)
                    new_entries_by_kind["improper"].append(entry)
                    main_entries.append(entry)
        return matches

    # Re-derive the type of every *existing* bond/angle/dihedral/improper
    # (i.e. one already present in the pre-reaction template, not one
    # newly formed by this reaction) that involves any atom whose type
    # was just changed by `type_adjustments` above -- so e.g. an
    # existing angle between a retyped atom and its unaffected neighbors
    # picks up whatever new angle-type parameters match its post-reaction
    # types, exactly as though the whole template had been generated
    # fresh with those types. Atom order within each entry is left
    # untouched (only its `type` label may change); if no matching
    # parameters are found for the new types, a warning is printed and
    # the entry's original type label is kept as-is.
    retyped_ids = {atom_id for atom_id, _ in (type_adjustments or [])}
    if retyped_ids:
        for kind, entries in (
            ("bond", template["bonds"]), ("angle", template["angles"]),
            ("dihedral", template["dihedrals"]), ("improper", template["impropers"]),
        ):
            for entry in entries:
                atoms = entry["atoms"]
                if not any(x in retyped_ids for x in atoms):
                    continue
                type_args = tuple(template["types"][x] for x in atoms)
                label = use_entry(kind, type_args)
                if label is None:
                    chain = "-".join(f"{x}({t})" for x, t in zip(atoms, type_args))
                    print(
                        f"Warning: no {kind} parameters found (in --params "
                        f"or --extra-params) for existing {kind} {chain} "
                        f"after posttype changes; keeping its original "
                        f"type '{entry['type']}'."
                    )
                    continue
                entry["type"] = label

    adjacency = rmol.build_adjacency(template["bonds"])

    ta, tb = template["types"][a], template["types"][b]
    found_bond_type = use_entry("bond", (ta, tb))
    if found_bond_type is None:
        sys.exit(
            f"Error: no bond parameters found (in --params or "
            f"--extra-params) for new bond {a}({ta})-{b}({tb})"
        )
    else:
        template["bonds"].append({"type": found_bond_type, "atoms": [a, b]})

    adjacency[a].append(b)
    adjacency[b].append(a)

    for u, va, vb in rmol.new_angles_for_bond(adjacency, a, b):
        t1, t2, t3 = template["types"][u], template["types"][va], template["types"][vb]
        label = use_entry("angle", (t1, t2, t3))
        if label is None:
            chain = f"{u}({t1})-{va}({t2})-{vb}({t3})"
            if verbose:
                print(
                    f"Warning: no angle parameters found (in --params or "
                    f"--extra-params) for new angle {chain}; "
                    f"omitting from output."
                )
            if t1 > t3:
                ttt = t1
                t1 = t3
                t3 = ttt
            missing["angle"].add(f"{t1}-{t2}-{t3}")
            continue
        template["angles"].append({"type": label, "atoms": [u, va, vb]})

    for quad in rmol.new_dihedrals_for_bond(adjacency, a, b):
        tt = tuple(template["types"][x] for x in quad)
        label = use_entry("dihedral", tt)
        if label is None:
            if verbose:
                chain = "-".join(f"{x}({t})" for x, t in zip(quad, tt))
                print(
                    f"Warning: no dihedral parameters found for new dihedral "
                    f"{chain}; omitting from output."
                )
            chain = "-".join(t for t in tt)
            chainr = "-".join(t for t in reversed(tt))
            if chainr < chain:
                chain = chainr
            missing["dihedral"].add(chain)
            continue
        template["dihedrals"].append({"type": label, "atoms": list(quad)})

    for center, wing_atoms in rmol.new_improper_centers_for_bond(adjacency, a, b):
        central_type = template["types"][center]
        wing_types = [template["types"][nb] for nb in wing_atoms]
        matches = use_improper_entries(central_type, wing_types)
        if not matches:
            if verbose:
                print(
                    f"Warning: no improper parameters found (in --params or "
                    f"--extra-params) for a new trigonal center at atom "
                    f"{center} ({central_type}) with substituents "
                    f"{'-'.join(wing_types)}; omitting."
                )
            missing["improper"].add(f"{central_type} {'-'.join(sorted(wing_types))}")
            continue
        for entry in matches:
            remaining = list(wing_atoms)
            ordered_atoms = [center]
            for wing_type in entry["types"][1:]:
                idx = next(
                    k for k, nb in enumerate(remaining)
                    if template["types"][nb] == wing_type
                )
                ordered_atoms.append(remaining.pop(idx))
            template["impropers"].append({"type": entry["label"], "atoms": ordered_atoms})

    if not nocharges:
        for atom_id, delta in charge_adjustments or []:
            if not delta:
                continue
            if atom_id not in template["charges"]:
                sys.exit(f"Error: atom {atom_id} not found in {pre_path}")
            template["charges"][atom_id] = (
                f"{float(template['charges'][atom_id]) + delta:.6f}"
            )

    rmol.write_molecule_template(
        template, post_path, known_edge_ids=known_edge_ids, nocharges=nocharges
    )
    if verbose:
        print(f"Wrote post-reaction molecule template to {post_path}")


def load_reactants_and_reactions(path):
    """Parse a --reactions-file: a `Reactants` section (lines of `name
    file initiatorID [charge Q0 Q1 ...] [posttype atomID:type ...]`)
    followed by a `Reactions` section (lines of `label reactant_a
    reactant_b`, where each reactant is either a Reactants name, for a
    fresh copy of that reactant, or `reaction_label:reactant_name`, for
    that reactant's own atom (in `reaction_label`'s whole post-reaction
    product) re-filtered by --max-bonds around it, just like a fresh
    copy -- see resolve_reactant_spec). Blank lines are ignored,
    and a `#` and everything after it on a line is treated as a comment
    and ignored.

    A Reactants line's optional `charge Q0 Q1 ... Qg ...` keyword gives,
    for each "generation" g of that reactant's initiatorID atom (g=0 the
    first time a fresh copy of it is used, not yet having formed any new
    bond via this file; g=1 after it has formed one new bond, whether as
    itself or via a `label:name` reference carrying it forward; and so
    on), the charge shift Qg to add to that atom's *original* (input
    file) charge once it has reached generation g. Qg values are
    absolute, not incremental: an atom at generation g always has charge
    `original + Qg`, regardless of the path taken to reach generation g.
    Q0 is also used to correct every fragment in the whole system
    matching this reactant's un-reacted (generation-0) shape (same atom
    typelabels/bonds out to --max-bonds, ignoring angles/dihedrals/
    impropers) when writing --new-data-file, since that file describes
    the system before any of these reactions have actually occurred.
    Reactants with no `charge` keyword are left entirely unmodified.

    A Reactants line's optional `posttype atomID:type atomID:type ...`
    keyword gives new atom types for one or more atoms of this reactant
    (each `atomID` in the reactant's own input-file, original numbering;
    `type` is either a LAMMPS atom type number or a type label, resolved
    against that same input file). These new types are applied *only* to
    this reactant's atoms in the post-reaction molecule template the
    first time a fresh copy of it reacts (the pre-reaction template, and
    any earlier/other reactions' templates, are left unchanged) -- e.g.
    to reflect a reaction that changes a substituent atom's hybridization/
    protonation state elsewhere on the same fragment as the new bond.
    Reactants with no `posttype` keyword are left entirely unmodified.

    `charge` and `posttype` may be given in either order on the same
    Reactants line, each consuming tokens up to the next keyword or end
    of line.

    Returns (reactants, reactions): reactants is {name: (file, atom_id,
    charges, posttypes)}, charges is a list of floats or None if no
    `charge` keyword was given, posttypes is a {atom_id: type_spec} dict
    (atom_id an int, type_spec a str) or None if no `posttype` keyword
    was given; reactions is an ordered list of (label, reactant_a,
    reactant_b).
    """
    reactants = {}
    reactions = []
    reaction_labels = set()
    section = None
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            low = line.lower()
            if low == "reactants":
                section = "reactants"
                continue
            if low == "reactions":
                section = "reactions"
                continue
            parts = line.split()
            if section == "reactants":
                if len(parts) < 3:
                    sys.exit(
                        f"{path}: line {lineno}: expected 'name file "
                        f"initiatorID [charge Q0 Q1 ...] "
                        f"[posttype atomID:type ...]', got: {line!r}"
                    )
                name, filename, atom_id = parts[0], parts[1], parts[2]
                charges = None
                posttypes = None
                idx = 3
                while idx < len(parts):
                    keyword = parts[idx].lower()
                    idx += 1
                    start = idx
                    while idx < len(parts) and parts[idx].lower() not in (
                        "charge", "posttype"
                    ):
                        idx += 1
                    values = parts[start:idx]
                    if not values:
                        sys.exit(
                            f"{path}: line {lineno}: '{keyword}' keyword "
                            f"needs at least one value"
                        )
                    if keyword == "charge":
                        if charges is not None:
                            sys.exit(
                                f"{path}: line {lineno}: 'charge' keyword "
                                f"given more than once"
                            )
                        try:
                            charges = [float(q) for q in values]
                        except ValueError:
                            sys.exit(
                                f"{path}: line {lineno}: 'charge' values "
                                f"must be numbers, got: {' '.join(values)!r}"
                            )
                    elif keyword == "posttype":
                        if posttypes is not None:
                            sys.exit(
                                f"{path}: line {lineno}: 'posttype' "
                                f"keyword given more than once"
                            )
                        posttypes = {}
                        for tok in values:
                            if ":" not in tok:
                                sys.exit(
                                    f"{path}: line {lineno}: 'posttype' "
                                    f"values must be 'atomID:type', got: "
                                    f"{tok!r}"
                                )
                            aid_str, type_spec = tok.split(":", 1)
                            try:
                                aid = int(aid_str)
                            except ValueError:
                                sys.exit(
                                    f"{path}: line {lineno}: 'posttype' "
                                    f"atomID must be an integer, got: "
                                    f"{aid_str!r}"
                                )
                            if not type_spec:
                                sys.exit(
                                    f"{path}: line {lineno}: 'posttype' "
                                    f"value {tok!r} is missing a type "
                                    f"after ':'"
                                )
                            if aid in posttypes:
                                sys.exit(
                                    f"{path}: line {lineno}: 'posttype' "
                                    f"given more than once for atom {aid}"
                                )
                            posttypes[aid] = type_spec
                    else:
                        sys.exit(
                            f"{path}: line {lineno}: expected 'name file "
                            f"initiatorID [charge Q0 Q1 ...] "
                            f"[posttype atomID:type ...]', got: {line!r}"
                        )
                if name in reactants:
                    sys.exit(
                        f"{path}: line {lineno}: duplicate reactant "
                        f"name '{name}'"
                    )
                try:
                    atom_id = int(atom_id)
                except ValueError:
                    sys.exit(
                        f"{path}: line {lineno}: initiatorID must be an "
                        f"integer, got {atom_id!r}"
                    )
                reactants[name] = (filename, atom_id, charges, posttypes)
            elif section == "reactions":
                if len(parts) != 3:
                    sys.exit(
                        f"{path}: line {lineno}: expected 'label "
                        f"reactant_a reactant_b', got: {line!r}"
                    )
                label, reactant_a, reactant_b = parts
                if label in reaction_labels:
                    sys.exit(
                        f"{path}: line {lineno}: duplicate reaction "
                        f"label '{label}'"
                    )
                reaction_labels.add(label)
                reactions.append((label, reactant_a, reactant_b))
            else:
                sys.exit(
                    f"{path}: line {lineno}: expected a 'Reactants' or "
                    f"'Reactions' heading before this line: {line!r}"
                )
    if not reactants:
        sys.exit(f"{path}: no 'Reactants' section found")
    if not reactions:
        sys.exit(f"{path}: no 'Reactions' section found")
    return reactants, reactions


def shift_atom_charge(sections, atom_id, delta):
    """Add `delta` to the charge (4th column) of the Atoms-section row
    for `atom_id` (in `sections`' own numbering), in place. No-op if
    `delta` is 0."""
    if not delta:
        return
    for row in sections.get("Atoms", []):
        if int(row[0]) == atom_id:
            row[3] = f"{float(row[3]) + delta:.6f}"
            return
    sys.exit(f"Error: atom {atom_id} not found while applying a charge shift")


def charge_delta(path, origin_name, charges, gen_before, gen_after):
    """The charge shift to apply going from generation `gen_before` to
    `gen_after` for a reactant whose Reactants-line `charge` list is
    `charges` (or None if it has none, in which case there's nothing to
    do). `charges[g]` is that atom's total shift once at generation g,
    so the shift to apply *now* is charges[gen_after] - charges[gen_before]
    (gen_before's own shift already having been applied earlier)."""
    if not charges:
        return 0.0
    if gen_after >= len(charges):
        sys.exit(
            f"{path}: reactant '{origin_name}' reaches generation "
            f"{gen_after} (has formed {gen_after} new bond(s)) but its "
            f"'charge' keyword only lists {len(charges)} value(s); add "
            f"at least {gen_after + 1} value(s)"
        )
    return charges[gen_after] - charges[gen_before]


def build_atom_type_and_adjacency(counts, sections):
    """Return (atom_type, adjacency): atom_type is {atom_id: label}
    (falling back to the raw type number, as a string, if no matching
    'Atom Type Labels' row exists) and adjacency is {atom_id: set of
    bonded atom_ids}, both from `sections`' own (Atoms/Bonds section)
    numbering."""
    atoms = sections.get("Atoms", [])
    type_labels = d2m.extract_label_map(sections, "atom")
    atom_type = {
        int(row[0]): type_labels.get(int(row[2]), row[2]) for row in atoms
    }
    adjacency = {int(row[0]): set() for row in atoms}
    for row in sections.get("Bonds", []):
        a1, a2 = int(row[2]), int(row[3])
        adjacency[a1].add(a2)
        adjacency[a2].add(a1)
    return atom_type, adjacency


def _bfs_order(adjacency, root):
    """Breadth-first traversal order of `adjacency` starting at `root`,
    assuming every node is reachable from it (true of any connected
    molecule fragment)."""
    order = []
    seen = {root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbor in sorted(adjacency[current]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return order


def _fragment_matches_at(pattern_order, pattern_type, pattern_adjacency, root_full,
                         full_type, full_adjacency):
    """Try to extend a mapping of pattern atoms -> full-graph atoms,
    rooted at pattern_order[0] -> root_full, to all of `pattern_order`,
    such that atom typelabels match and the induced subgraph (bonds only,
    among the mapped atoms) is exactly isomorphic. Returns True/False."""
    mapping = {pattern_order[0]: root_full}
    used = {root_full}

    def extend(idx):
        if idx == len(pattern_order):
            return True
        p = pattern_order[idx]
        anchor = next((q for q in pattern_adjacency[p] if q in mapping), None)
        if anchor is None:
            return False
        for candidate in full_adjacency.get(mapping[anchor], ()):
            if candidate in used or full_type.get(candidate) != pattern_type[p]:
                continue
            if any(
                q in mapping and mapping[q] not in full_adjacency.get(candidate, ())
                for q in pattern_adjacency[p]
            ):
                continue
            if any(
                fq in full_adjacency.get(candidate, ()) and q not in pattern_adjacency[p]
                for q, fq in mapping.items()
            ):
                continue
            mapping[p] = candidate
            used.add(candidate)
            if extend(idx + 1):
                return True
            del mapping[p]
            used.discard(candidate)
        return False

    return extend(1)


def find_fragment_matches(pattern_center, pattern_type, pattern_adjacency,
                           full_type, full_adjacency):
    """Return the sorted list of every atom (full-graph numbering) whose
    neighborhood, out to the same distance as the pattern fragment, is
    isomorphic to it (matching atom typelabels and bonds; angles/
    dihedrals/impropers are ignored), rooted at that atom the same way
    `pattern_center` roots the pattern fragment."""
    order = _bfs_order(pattern_adjacency, pattern_center)
    root_label = pattern_type[pattern_center]
    matches = [
        atom for atom, label in full_type.items()
        if label == root_label
        and _fragment_matches_at(order, pattern_type, pattern_adjacency, atom,
                                  full_type, full_adjacency)
    ]
    return sorted(matches)


def build_data_file_charge_edits(data_file, reactants, max_bonds):
    """For every Reactants entry with a `charge` keyword, find every
    fragment in `data_file` matching its generation-0 (un-reacted) shape
    (same atom typelabels/bonds out to `max_bonds`, ignoring angles/
    dihedrals/impropers), and record a `original + charges[0]` charge
    edit for each match's own initiatorID-equivalent atom.

    Returns {atom_id: new_charge} (data_file's own atom numbering).
    """
    full_counts, full_sections, extra_lines = d2m.parse_fragment_file(data_file)
    full_type, full_adjacency = build_atom_type_and_adjacency(full_counts, full_sections)
    full_charges = {
        int(row[0]): float(row[3]) for row in full_sections.get("Atoms", [])
    }

    edits = {}
    for name, (filename, atom_id, charges, _posttypes) in reactants.items():
        if not charges:
            continue
        pattern_counts, pattern_sections, pattern_center, _, _ = parse_and_filter_fragment(
            filename, atom_id, max_bonds
        )
        pattern_type, pattern_adjacency = build_atom_type_and_adjacency(
            pattern_counts, pattern_sections
        )
        matches = find_fragment_matches(
            pattern_center, pattern_type, pattern_adjacency, full_type, full_adjacency
        )
        if not matches:
            print(
                f"Warning: reactant '{name}' has a 'charge' keyword but "
                f"no atom in {data_file} matches its fragment shape; no "
                "charges were changed for it."
            )
            continue
        print(
            f"Reactant '{name}': found {len(matches)} matching fragment(s) "
            f"in {data_file}; shifting each by {charges[0]:+g}"
        )
        for match_id in matches:
            new_charge = full_charges[match_id] + charges[0]
            if match_id in edits and edits[match_id] != new_charge:
                print(
                    f"Warning: atom {match_id} in {data_file} matches "
                    "more than one reactant's fragment shape with "
                    "different charge shifts; using the last one applied."
                )
            edits[match_id] = new_charge
    return edits


def apply_charge_edits_to_data_file(path, edits):
    """Rewrite `path`'s Atoms section in place, replacing the charge
    (4th column) of every atom_id in `edits` with its new value."""
    if not edits:
        return
    with open(path) as f:
        lines = f.readlines()

    start = end = None
    for i, raw in enumerate(lines):
        if raw.split("#", 1)[0].strip() == "Atoms":
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            start = j
            k = start
            while k < len(lines) and lines[k].strip() != "":
                k += 1
            end = k
            break
    if start is None:
        sys.exit(f"Error: no 'Atoms' section found in {path}")

    for i in range(start, end):
        body, sep, comment = lines[i].partition("#")
        tokens = body.split()
        if not tokens:
            continue
        atom_id = int(tokens[0])
        if atom_id in edits:
            tokens[3] = f"{edits[atom_id]:.6f}"
            new_line = " ".join(tokens)
            if sep:
                new_line += " #" + comment.rstrip("\n")
            lines[i] = new_line + "\n"

    with open(path, "w") as f:
        f.writelines(lines)


def resolve_posttype_local_adjustments(path, spec, filename, atom_id, posttypes, counts, sections, id_map):
    """Translate a reactant's `posttype` dict ({original atom_id: type
    spec}, in `filename`'s own numbering) into a list of (local_atom_id,
    new_type_label) pairs, `local_atom_id` in the filtered fragment's own
    (post `parse_and_filter_fragment`) numbering, via `id_map`.

    `type spec` is resolved against `filename` itself (via `counts`/
    `sections`, which -- since fragment-filtering leaves type
    information untouched -- still describe the *whole* file's atom
    types, not just the filtered fragment): either a plain integer atom
    type number (must be within `filename`'s declared range of atom
    types) or a type label (must appear in `filename`'s own 'Atom Type
    Labels' section). Exits with an error message if a type isn't
    present in `filename`, or if an `atomID` isn't among the fragment's
    kept atoms (i.e. not present in `filename` at all, or filtered out
    by --max-bonds).
    """
    if not posttypes:
        return []

    atom_type_label_map = d2m.extract_label_map(sections, "atom")
    label_to_num = {label: num for num, label in atom_type_label_map.items()}
    n_atom_types = counts.get("atom types", 0)

    adjustments = []
    for orig_id, type_spec in posttypes.items():
        if orig_id not in id_map:
            sys.exit(
                f"{path}: reactant '{spec}': posttype atom {orig_id} is "
                f"not present in {filename}'s molecule template (it "
                f"either doesn't exist in {filename}, or was filtered "
                f"out by --max-bonds around atom {atom_id})"
            )
        if type_spec.isdigit():
            type_num = int(type_spec)
            if type_num < 1 or type_num > n_atom_types:
                sys.exit(
                    f"{path}: reactant '{spec}': posttype type "
                    f"'{type_spec}' for atom {orig_id} is not a valid "
                    f"atom type in {filename} (it declares {n_atom_types} "
                    f"atom type(s))"
                )
            new_label = atom_type_label_map.get(type_num, str(type_num))
        else:
            if type_spec not in label_to_num:
                sys.exit(
                    f"{path}: reactant '{spec}': posttype type label "
                    f"'{type_spec}' for atom {orig_id} was not found in "
                    f"{filename}'s 'Atom Type Labels' section"
                )
            new_label = type_spec
        adjustments.append((id_map[orig_id], new_label))
    return adjustments


def resolve_reactant_spec(
    path, spec, current_label, reactants, max_bonds, initiator_locations, post_path_of,
    bag_of, generation_at, redirect_label=None,
):
    """Resolve one reaction line's reactant spec (either a Reactants
    name, for a fresh copy of that reactant's file, or
    `reaction_label:reactant_name`, for the fresh atoms `reactant_name`
    contributed to an earlier reaction's whole product, re-filtered by
    `max_bonds` around that atom just like a fresh copy).

    `redirect_label` (optional), if given, is a dict mapping a reaction
    label that --skip-equivalent caused to be skipped to the earlier,
    equivalent reaction label whose product should be used instead --
    so `reaction_label:reactant_name` specs referencing a skipped
    reaction transparently use that earlier reaction's product.

    Returns ((counts, sections, center_new_id, edge_new_ids), signature,
    charges, generation, posttype_local) where the first element is the
    same fragment tuple shape as parse_and_filter_fragment (already
    carrying, on a fresh copy, its generation-0 charge shift baked into
    its initiatorID atom's charge -- see the `charge` keyword docs on
    load_reactants_and_reactions), `signature` is (origin_name, bag): the
    original Reactants name this fragment's bonding atom came from, and
    a sorted tuple of every Reactants name incorporated into this
    fragment (with repeats, so multiplicity is preserved) -- together
    identifying this reactant well enough to detect equivalent reactions
    (see reactant_pair_signature); `charges` is that Reactants name's
    charge list (or None); `generation` is how many new bonds this
    specific initiatorID atom has already formed via this file, before
    this reaction (0 for a fresh copy); and `posttype_local` is a list of
    (local_atom_id, new_type_label) pairs (local to this fragment, i.e.
    before any merge-offset is applied) from that Reactants name's
    `posttype` keyword -- empty unless this is a fresh copy of a
    reactant with a `posttype` keyword (a `label:name` back-reference to
    an already-reacted reactant never re-applies its `posttype`).
    """
    if ":" in spec:
        ref_label, origin_name = spec.split(":", 1)
        if redirect_label is not None:
            seen_redirects = set()
            while ref_label in redirect_label:
                if ref_label in seen_redirects:
                    break  # defend against a cycle; shouldn't happen
                seen_redirects.add(ref_label)
                ref_label = redirect_label[ref_label]
        if ref_label not in post_path_of:
            sys.exit(
                f"{path}: reaction '{current_label}': unknown reaction "
                f"label '{ref_label}' referenced in '{spec}' (it must "
                f"be an earlier reaction's label)"
            )
        site = initiator_locations[ref_label].get(origin_name)
        if site is None:
            sys.exit(
                f"{path}: reaction '{current_label}': '{origin_name}' is "
                f"not an available initiator site in reaction "
                f"'{ref_label}''s product (used in '{spec}')"
            )
        post_path = post_path_of[ref_label]
        counts, sections = d2m.parse_fragment_file(post_path)
        try:
            counts, sections, center_new_id, edge_new_ids, _id_map = (
                d2m.filter_by_distance(counts, sections, site, max_bonds)
            )
        except ValueError as exc:
            sys.exit(
                f"{path}: reaction '{current_label}': '{spec}': {post_path}: "
                f"{exc}"
            )
        _, _, charges, _posttypes = reactants[origin_name]
        generation = generation_at[ref_label][origin_name]
        return (
            (counts, sections, center_new_id, edge_new_ids),
            (origin_name, bag_of[ref_label]), charges, generation, [],
        )

    if spec not in reactants:
        sys.exit(
            f"{path}: reaction '{current_label}': unknown reactant "
            f"'{spec}' (not in the 'Reactants' section)"
        )
    filename, atom_id, charges, posttypes = reactants[spec]
    counts, sections, center_new_id, edge_new_ids, id_map = parse_and_filter_fragment(
        filename, atom_id, max_bonds, extra_centers=sorted(posttypes) if posttypes else ()
    )
    if charges:
        shift_atom_charge(sections, center_new_id, charges[0])
    posttype_local = resolve_posttype_local_adjustments(
        path, spec, filename, atom_id, posttypes, counts, sections, id_map
    )
    return (
        (counts, sections, center_new_id, edge_new_ids), (spec, (spec,)),
        charges, 0, posttype_local,
    )


def reactant_pair_signature(sig_a, sig_b):
    """A hashable signature for a reaction's whole pair of reactants,
    order-independent (so e.g. Zn+IM1 and IM1+Zn compare equal), used to
    detect reactions equivalent to an earlier one."""
    return frozenset((sig_a, sig_b))


def reactant_pair_structural_signature(fragment_a, fragment_b):
    """A hashable signature for a reaction's whole pair of *filtered*
    reactant fragments, order-independent, based on each fragment's
    d2m.fragment_canonical_signature rather than the Reactants names
    that produced it. Unlike reactant_pair_signature (which only
    catches the same names combined the same way), this also catches
    two differently-named/differently-derived reactants whose --max-
    bonds-filtered neighborhoods turned out to be the same topology
    (e.g. a growing product where the one ligand differing between two
    histories got entirely trimmed away by the distance filter)."""
    struct_sig_a = d2m.fragment_canonical_signature(*fragment_a)
    struct_sig_b = d2m.fragment_canonical_signature(*fragment_b)
    return frozenset((struct_sig_a, struct_sig_b))


def record_molecule_lines(
    molecule_lines, react_lines, reaction_name, label, pre_path, post_path, map_path,
):
    """If --lammps-reaction-input is in use, append this reaction's
    `molecule moleculeID molecule_file` lines (one per pre/post-reaction
    template file) to `molecule_lines`, and a corresponding `react ...` clause
    (for use inside a `fix bond/react` command) to `react_lines`."""
    if molecule_lines is None:
        return
    pre_id = f"{reaction_name}_{label}_pre"
    post_id = f"{reaction_name}_{label}_post"
    molecule_lines.append(f"molecule {pre_id} {pre_path}")
    molecule_lines.append(f"molecule {post_id} {post_path}")
    react_lines.append(
        f"react {reaction_name}_{label} react-group-ID Nevery Rmin Rmax "
        f"{pre_id} {post_id} {map_path}"
    )


def run_reactions_file(
    args, all_entries, main_entries, added_keys, new_entries_by_kind, molecule_lines,
    react_lines, missing, equivalencies
):
    reactants, reactions = load_reactants_and_reactions(args.reactions_file)

    initiator_locations = {}  # label -> {reactant_name: atom_id in that label's product}
    post_path_of = {}  # label -> post-reaction template path
    bag_of = {}  # label -> frozenset of Reactants names incorporated into its product
    seen_pair_sigs = {}  # reactant_pair_signature -> label of the first reaction with it
    seen_struct_pair_sigs = {}  # reactant_pair_structural_signature -> label of the first
    generation_at = {}  # label -> {reactant_name: bonds formed so far via this file}

    for label, reactant_a, reactant_b in reactions:
        pre_path = f"{args.reaction_name}_{label}-pre.molecule_template"
        post_path = f"{args.reaction_name}_{label}-post.molecule_template"
        map_path = f"{args.reaction_name}_{label}.map"

        fragment_a, sig_a, charges_a, gen_before_a, posttype_local_a = resolve_reactant_spec(
            args.reactions_file, reactant_a, label, reactants, args.max_bonds,
            initiator_locations, post_path_of, bag_of, generation_at, equivalencies,
        )
        fragment_b, sig_b, charges_b, gen_before_b, posttype_local_b = resolve_reactant_spec(
            args.reactions_file, reactant_b, label, reactants, args.max_bonds,
            initiator_locations, post_path_of, bag_of, generation_at, equivalencies,
        )

        # This check is faster than the next one and operates on just the reactants
        # it assumes that ABC and CBA are the equivalent
        pair_sig = reactant_pair_signature(sig_a, sig_b)
        if pair_sig in seen_pair_sigs:
            if args.skip_equivalent:
                req = seen_pair_sigs[pair_sig]
                equivalencies[label] = req
                if req in equivalencies:
                    equivalencies[label] = equivalencies[req]
                continue
            sys.exit(
                f"{args.reactions_file}: reaction '{label}' is "
                f"equivalent to reaction '{seen_pair_sigs[pair_sig]}' "
                f"(same reactants and initiators, ignoring which side "
                f"each is given on)"
            )
        else:
            seen_pair_sigs[pair_sig] = label

        # This check is the check we actually want.  It's just more expensive
        # than the first check.
        struct_pair_sig = reactant_pair_structural_signature(fragment_a, fragment_b)
        if struct_pair_sig in seen_struct_pair_sigs:
            if args.skip_equivalent:
                req = seen_struct_pair_sigs[struct_pair_sig]
                equivalencies[label] = req
                if req in equivalencies:
                    equivalencies[label] = equivalencies[req]
                continue
            sys.exit(
                f"{args.reactions_file}: reaction '{label}' is "
                f"equivalent to reaction "
                f"'{seen_struct_pair_sigs[struct_pair_sig]}' -- after "
                f"--max-bonds filtering, its two reactants' neighborhoods "
                f"have the same topology (atom types and bond graph), "
                f"even though they were named/derived differently"
            )
        else:
            seen_struct_pair_sigs[struct_pair_sig] = label

        counts, atom_offsets, offset_ids, final_edge_ids = build_and_write_pre_template(
            [fragment_a, fragment_b], pre_path, nocharges=args.nocharges, verbose=args.verbose
        )
        bond = (offset_ids[0], offset_ids[1])

        d2m.write_map_file(map_path, counts["atoms"], list(bond), final_edge_ids)
        if args.verbose:
            print(f"Wrote bond/react map file to {map_path}")

        gen_after_a, gen_after_b = gen_before_a + 1, gen_before_b + 1
        charge_adjustments = [
            (bond[0], charge_delta(args.reactions_file, sig_a[0], charges_a, gen_before_a, gen_after_a)),
            (bond[1], charge_delta(args.reactions_file, sig_b[0], charges_b, gen_before_b, gen_after_b)),
        ]
        type_adjustments = [
            (local_id + atom_offsets[0], new_label)
            for local_id, new_label in posttype_local_a
        ] + [
            (local_id + atom_offsets[1], new_label)
            for local_id, new_label in posttype_local_b
        ]

        react_pre_template(
            pre_path, post_path, map_path, bond, all_entries,
            main_entries, added_keys, new_entries_by_kind, missing, charge_adjustments,
            type_adjustments=type_adjustments, nocharges=args.nocharges, verbose=args.verbose
        )
        record_molecule_lines(
            molecule_lines, react_lines, args.reaction_name, label, pre_path, post_path,
            map_path,
        )

        origin_a, origin_b = sig_a[0], sig_b[0]
        initiator_locations[label] = {origin_a: offset_ids[0], origin_b: offset_ids[1]}
        bag_of[label] = tuple(sorted(sig_a[1] + sig_b[1]))
        post_path_of[label] = post_path
        generation_at[label] = {origin_a: gen_after_a, origin_b: gen_after_b}

    if args.data_file is not None:
        return build_data_file_charge_edits(args.data_file, reactants, args.max_bonds)
    return {}

class d2r_args:
    def __init__(self, reaction_name, reactions_file, params, extra_params, merged_params,
                 data_file, new_data_file, lammps_reaction_input):
        self.reaction_name = reaction_name
        self.reactions_file = reactions_file
        self.max_bonds = 3
        self.nocharges = True
        self.skip_equivalent = True
        self.params = params # list of files
        self.extra_params = extra_params
        self.merged_params = merged_params
        self.data_file = data_file
        self.new_data_file = new_data_file
        self.lammps_reaction_input = lammps_reaction_input
        self.verbose = False

def main():
    args = parse_args()
    data2reaction(args)

def data2reaction(args):
    base_main_entries = [e for p in args.params for e in rmol.parse_params_file(p)]
    extra_entries = [e for p in args.extra_params for e in rmol.parse_params_file(p)]
    all_entries = base_main_entries + extra_entries
    # main_entries starts as a *copy* of base_main_entries and grows as
    # types are discovered across the sequence, so later reactions
    # recognize types found by earlier ones; base_main_entries itself
    # (used for final ID assignment) is never mutated.
    main_entries = list(base_main_entries)
    added_keys = set()
    new_entries_by_kind = {"bond": [], "angle": [], "dihedral": [], "improper": []}
    molecule_lines = [] if args.lammps_reaction_input is not None else None
    react_lines = [] if args.lammps_reaction_input is not None else None

    missing = {"angle": set(), "dihedral": set(), "improper": set()}
    equivalencies = {}
    data_file_charge_edits = run_reactions_file(
        args, all_entries, main_entries, added_keys, new_entries_by_kind, molecule_lines,
        react_lines, missing, equivalencies
    )
    if args.verbose:
        print("Summary:")
    for t, m in missing.items():
        if len(m) > 0:
            m = sorted(list(m))
            print("Missing", t, "parameters for")
            print("", "\n ".join(m))
    if len(equivalencies) > 0:
        print(len(equivalencies), "reactions skipped due to equivalence")
        if args.verbose:
            equivLists = {}
            for r1, req in equivalencies.items():
                if req not in equivLists:
                    equivLists[req] = []
                equivLists[req].append(r1)
            for req, eqList in equivLists.items():
                print(" ", req, " ", " ".join(eqList))

    if args.lammps_reaction_input is not None:
        fix_lines = ["fix ID group-ID bond/react &"] + [
            f"  {line}" + (" &" if k < len(react_lines) - 1 else "")
            for k, line in enumerate(react_lines)
        ]
        with open(args.lammps_reaction_input, "w") as f:
            f.write("\n".join(molecule_lines) + "\n\n")
            f.write("\n".join(fix_lines) + "\n")
        print(
            f"Wrote LAMMPS reaction input to {args.lammps_reaction_input} "
            f"({len(molecule_lines)} molecule line(s), {len(react_lines)} react clause(s))"
        )

    if args.merged_params is not None or args.data_file is not None:
        assigned = rmol.assign_new_ids(new_entries_by_kind, base_main_entries)

    if args.merged_params is not None:
        rmol.write_merged_params(args.params, assigned, args.merged_params)
        n_new = sum(len(v) for v in assigned.values())
        print(f"Wrote merged parameter file to {args.merged_params} ({n_new} new type(s))")

    if args.data_file is not None:
        rmol.write_updated_data_file(args.data_file, args.new_data_file, assigned)
        n_new = sum(len(v) for v in assigned.values())
        print(
            f"Wrote updated data file to {args.new_data_file} ({n_new} new "
            "type(s) added to Type Labels sections)"
        )
        if data_file_charge_edits:
            apply_charge_edits_to_data_file(args.new_data_file, data_file_charge_edits)
            print(
                f"Updated {len(data_file_charge_edits)} atom charge(s) in "
                f"{args.new_data_file}"
            )


if __name__ == "__main__":
    main()
