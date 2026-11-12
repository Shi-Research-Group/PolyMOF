#!/usr/bin/env python3
"""
transfer_prmtop_charges.py

Transfer partial charges from one or more AMBER .prmtop files whose
charges are already known ("old" molecules) onto a second .prmtop file
describing a related but not-necessarily-identical molecule ("new"
molecule) whose charges have not yet been determined (e.g. still 0.0,
or otherwise placeholders), and write out a new .prmtop file for the
new molecule with those charges filled in.

An atom in the new molecule is considered "equivalent" to an atom in
an old molecule -- and so its charge contributes to that atom's charge
-- if the local fragment centered on it (every atom within --max-bonds
bonds, the user-specified fragment radius) has the same canonical
signature as the old atom's fragment. This is exactly the same
renumbering-invariant fragment/equivalence test used by
data2molecule.py's `fragment_canonical_signature` (and reused by
data2reaction.py and find_unique_fragments.py): a fragment is defined
by the AMBER atom type of every atom out to --max-bonds bonds and how
they're bonded to each other (1-dimensional Weisfeiler-Leman color
refinement), which is sufficient since angles/dihedrals/impropers are
fully determined by atom types and bonds and add no further
information.

When more than one old .prmtop file is given, every one of them is
searched for equivalent atoms, and the new atom's charge is set to the
average charge across *all* matching atoms found in *any* of them
(e.g. a repeat unit's charge averaged over every old molecule/
conformer it appears equivalently in, not just the first old file that
happens to contain a match).

If no equivalent atom can be found in any old molecule for some atom
in the new molecule, that atom's charge is left unchanged (i.e. kept
exactly as it already appears in the new .prmtop file), and a warning
is printed identifying it by atom number and AMBER atom type.

Because charges are copied/left unchanged atom-by-atom, the resulting
total (net) charge of the new molecule can come out non-integral/non-
physical (e.g. if some atoms couldn't be matched and were left at
0.0, or matched atoms' charges just don't happen to add up cleanly).
By default this script corrects for that: after copying, every atom's
charge is shifted by the same small constant amount so the new
molecule's net charge exactly equals the (average, across all old
files) old molecules' own net charge (rounded to the nearest integer).
Use --no-neutralize to skip this.

Usage:
    python3 transfer_prmtop_charges.py old.prmtop new.prmtop output.prmtop
    python3 transfer_prmtop_charges.py old.prmtop new.prmtop output.prmtop --max-bonds 4
    python3 transfer_prmtop_charges.py old1.prmtop old2.prmtop old3.prmtop new.prmtop output.prmtop
"""

import sys
import re
import argparse
from collections import deque


FORMAT_RE = re.compile(r"^%FORMAT\(\s*(\d+)([a-zA-Z])(\d+)(?:\.(\d+))?\s*\)\s*$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer partial charges from one or more AMBER .prmtop "
        "files with known charges onto a related .prmtop file, matching "
        "atoms by local bonded-fragment signature (same equivalence test "
        "used by data2molecule.py / data2reaction.py / "
        "find_unique_fragments.py)."
    )
    parser.add_argument(
        "old_prmtops", nargs="+", metavar="old_prmtop",
        help="One or more prmtop file(s) with known/good charges. Every "
        "one of them is searched for equivalent atoms; if more than one "
        "contains a match for a given new-molecule atom, that atom's "
        "charge is the average across all of them.",
    )
    parser.add_argument("new_prmtop", help="prmtop file whose charges should be filled in.")
    parser.add_argument("output_prmtop", help="Output prmtop file to write (a copy of new_prmtop with charges updated).")
    parser.add_argument(
        "--max-bonds", dest="max_bonds", type=int, default=3, metavar="N",
        help="Fragment radius, in bonds, used to determine whether an atom "
        "in the new molecule is equivalent to one in the old molecule. "
        "Default: 3.",
    )
    parser.add_argument(
        "--no-neutralize", dest="neutralize", action="store_false",
        help="Skip net-charge correction: by default, after charges are "
        "copied/left unchanged, the output charges are shifted by an "
        "equal small amount on every atom so their sum exactly matches "
        "the old molecule's own total charge (rounded to the nearest "
        "integer), since copying charges atom-by-atom (and leaving any "
        "unmatched atoms at their existing value) can otherwise leave "
        "the new molecule with a non-physical net charge.",
    )
    args = parser.parse_args()
    if args.max_bonds < 0:
        parser.error("--max-bonds must be >= 0")
    return args


class PrmtopFile:
    """A parsed AMBER prmtop file: keeps the raw lines (so unrelated
    sections can be written back out byte-for-byte unchanged) plus an
    index of %FLAG sections (name -> (format_line_index, first_data_line_index,
    last_data_line_index_exclusive, (count_per_line, ftype, width, decimals))).
    """

    def __init__(self, path):
        with open(path) as f:
            # keep line endings so the file round-trips exactly except
            # where we deliberately overwrite the CHARGE section
            self.lines = f.readlines()
        self.sections = {}
        i = 0
        n = len(self.lines)
        while i < n:
            line = self.lines[i].rstrip("\n")
            if line.startswith("%FLAG"):
                name = line[5:].strip()
                fmt_idx = i + 1
                fmt_line = self.lines[fmt_idx].rstrip("\n") if fmt_idx < n else ""
                m = FORMAT_RE.match(fmt_line.strip())
                if not m:
                    raise ValueError(
                        f"Expected %FORMAT(...) after %FLAG {name}, got: {fmt_line!r}"
                    )
                count_per_line = int(m.group(1))
                ftype = m.group(2).upper()
                width = int(m.group(3))
                decimals = int(m.group(4)) if m.group(4) else 0
                data_start = fmt_idx + 1
                j = data_start
                while j < n and not self.lines[j].startswith("%"):
                    j += 1
                self.sections[name] = (fmt_idx, data_start, j, (count_per_line, ftype, width, decimals))
                i = j
            else:
                i += 1

    def _iter_fields(self, data_start, data_end, width):
        for k in range(data_start, data_end):
            line = self.lines[k].rstrip("\n")
            for c in range(0, len(line), width):
                field = line[c:c + width]
                if field.strip() == "" and c + width >= len(line):
                    continue
                yield field

    def get_values(self, name):
        """Return the parsed values (list of str/int/float, per the
        section's own FORMAT type) for %FLAG `name`."""
        fmt_idx, data_start, data_end, (count_per_line, ftype, width, decimals) = self.sections[name]
        values = []
        for field in self._iter_fields(data_start, data_end, width):
            text = field.strip()
            if text == "":
                continue
            if ftype == "I":
                values.append(int(text))
            elif ftype in ("E", "F", "D"):
                values.append(float(text.replace("D", "E").replace("d", "e")))
            else:  # 'a'/'A' -> string
                values.append(text)
        return values

    def format_float_field(self, value, width, decimals):
        """Format `value` the way Fortran's E<width>.<decimals> would,
        e.g. width=16, decimals=8 -> ' -1.40311710E+00'."""
        text = f"{value:.{decimals}E}"
        # Python already gives a 2-digit exponent for normal-magnitude
        # doubles (e.g. 'E+00'), matching Fortran's default here.
        return text.rjust(width)

    def replace_section_values(self, name, new_values):
        """Rewrite the data lines of %FLAG `name` with `new_values`,
        re-using that section's own (count_per_line, ftype, width,
        decimals), and update self.lines/self.sections in place."""
        fmt_idx, data_start, data_end, (count_per_line, ftype, width, decimals) = self.sections[name]
        new_lines = []
        for k in range(0, len(new_values), count_per_line):
            chunk = new_values[k:k + count_per_line]
            if ftype in ("E", "F", "D"):
                fields = [self.format_float_field(v, width, decimals) for v in chunk]
            elif ftype == "I":
                fields = [str(v).rjust(width) for v in chunk]
            else:
                fields = [str(v).ljust(width) for v in chunk]
            new_lines.append("".join(fields) + "\n")
        old_n_lines = data_end - data_start
        self.lines[data_start:data_end] = new_lines
        shift = len(new_lines) - old_n_lines
        self.sections[name] = (fmt_idx, data_start, data_start + len(new_lines), (count_per_line, ftype, width, decimals))
        if shift:
            for other, (o_fmt, o_start, o_end, o_meta) in self.sections.items():
                if other != name and o_start > data_start:
                    self.sections[other] = (o_fmt + shift, o_start + shift, o_end + shift, o_meta)

    def write(self, path):
        with open(path, "w") as f:
            f.writelines(self.lines)


def build_adjacency(prm):
    """Return {atom_id (1-based): set(bonded atom_ids)} built from both
    BONDS_INC_HYDROGEN and BONDS_WITHOUT_HYDROGEN. Each bond entry is
    (ib, jb, icb) where ib/jb are 3x the 0-based atom index."""
    n_atoms = len(prm.get_values("AMBER_ATOM_TYPE"))
    adjacency = {aid: set() for aid in range(1, n_atoms + 1)}
    for section in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        if section not in prm.sections:
            continue
        values = prm.get_values(section)
        for k in range(0, len(values), 3):
            ib, jb, icb = values[k], values[k + 1], values[k + 2]
            a1 = ib // 3 + 1
            a2 = jb // 3 + 1
            adjacency[a1].add(a2)
            adjacency[a2].add(a1)
    return adjacency


def fragment_signature(adjacency, type_label, root, max_bonds):
    """Canonical (renumbering-invariant) signature of the fragment
    centered on `root` (every atom within `max_bonds` bonds), using the
    same 1-D Weisfeiler-Leman color-refinement equivalence test as
    data2molecule.py's fragment_canonical_signature: every atom starts
    colored by (its AMBER atom type, whether it's a boundary/"edge"
    atom whose topology got truncated by max_bonds, whether it's the
    root), then is repeatedly recolored by (its color, sorted multiset
    of its bonded neighbors' colors) until colors stop refining further.
    """
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

    keep_ids = set(distances)
    edge_ids = {
        aid for aid in keep_ids
        if any(nb not in keep_ids for nb in adjacency[aid])
    }
    sub_adjacency = {aid: (adjacency[aid] & keep_ids) for aid in keep_ids}
    n_bonds = sum(len(nbs) for nbs in sub_adjacency.values()) // 2

    color = {
        aid: (type_label[aid], aid in edge_ids, aid == root)
        for aid in keep_ids
    }
    prev_n_classes = 0
    for _ in range(len(keep_ids) + 1):
        color = {
            aid: (color[aid], tuple(sorted(color[nb] for nb in sub_adjacency[aid])))
            for aid in keep_ids
        }
        n_classes = len(set(color.values()))
        if n_classes == prev_n_classes:
            break
        prev_n_classes = n_classes

    return (
        len(keep_ids), n_bonds, color[root], tuple(sorted(color.values())),
    )


def main():
    args = parse_args()

    new_prm = PrmtopFile(args.new_prmtop)
    new_types_raw = new_prm.get_values("AMBER_ATOM_TYPE")
    new_names = new_prm.get_values("ATOM_NAME")
    new_charges = new_prm.get_values("CHARGE")
    new_type_label = {i + 1: t for i, t in enumerate(new_types_raw)}
    new_adjacency = build_adjacency(new_prm)

    # Precompute every old-molecule atom's fragment signature, across
    # *all* old prmtop files, as sig -> list of (old_prmtop_path, charge)
    # for every atom (in any old file) whose fragment has that signature.
    sig_to_matches = {}
    old_net_charges = []
    for old_path in args.old_prmtops:
        old_prm = PrmtopFile(old_path)
        old_types_raw = old_prm.get_values("AMBER_ATOM_TYPE")
        old_charges = old_prm.get_values("CHARGE")
        old_type_label = {i + 1: t for i, t in enumerate(old_types_raw)}
        old_adjacency = build_adjacency(old_prm)
        old_net_charges.append(sum(old_charges))

        for old_id in old_type_label:
            sig = fragment_signature(old_adjacency, old_type_label, old_id, args.max_bonds)
            sig_to_matches.setdefault(sig, []).append((old_path, old_charges[old_id - 1]))

    # When a signature matches more than one old atom -- whether several
    # equivalent positions within one old molecule (e.g. repeated units
    # in a polymer) or equivalent atoms found across multiple old
    # molecules/conformers -- use the average of all their charges
    # rather than picking one arbitrarily.
    sig_to_charge = {
        sig: sum(charge for _path, charge in matches) / len(matches)
        for sig, matches in sig_to_matches.items()
    }

    ambiguous_groups = [matches for matches in sig_to_matches.values() if len(matches) > 1]
    if ambiguous_groups:
        n_ambiguous_atoms = sum(len(matches) for matches in ambiguous_groups)
        old_desc = (
            args.old_prmtops[0] if len(args.old_prmtops) == 1
            else f"the {len(args.old_prmtops)} old prmtop file(s)"
        )
        print(
            f"Note: {len(sig_to_matches)} fragment(s) with {n_ambiguous_atoms} total duplicates across {old_desc}.  Averaging charges."
        )

    n_matched = 0
    n_unmatched = 0
    output_charges = list(new_charges)
    for new_id in new_type_label:
        sig = fragment_signature(new_adjacency, new_type_label, new_id, args.max_bonds)
        if sig in sig_to_charge:
            output_charges[new_id - 1] = sig_to_charge[sig]
            n_matched += 1
        else:
            n_unmatched += 1
            print(
                f"No equivalent atom found for new atom {new_id} "
                f"(name {new_names[new_id - 1]}, type {new_types_raw[new_id - 1]}); "
                f"leaving its charge unchanged ({new_charges[new_id - 1]:.8f})."
            )

    raw_net_charge = sum(output_charges)
    if args.neutralize:
        target_net_charge = round(sum(old_net_charges) / len(old_net_charges))
        residual = target_net_charge - raw_net_charge
        per_atom_shift = residual / len(output_charges)
        output_charges = [c + per_atom_shift for c in output_charges]
        print(
            f"Net charge before correction: {raw_net_charge:.8f}; old molecule(s)' "
            f"(average, rounded) net charge: {target_net_charge}; shifting every "
            f"atom by {per_atom_shift:+.8f} so the output's net charge is exactly "
            f"{target_net_charge}."
        )
    else:
        print(f"Net charge (no correction applied): {raw_net_charge:.8f}.")

    new_prm.replace_section_values("CHARGE", output_charges)
    new_prm.write(args.output_prmtop)

    strout = f"Matched {n_matched}/{len(new_type_label)} atom(s); "
    if n_unmatched > 0:
        strout += f"{n_unmatched} atom(s) left with their original charge. "
    strout += f"Wrote {args.output_prmtop}."
    print(strout)


if __name__ == "__main__":
    main()
