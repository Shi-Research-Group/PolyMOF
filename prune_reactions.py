#!/usr/bin/env python3
"""
prune_reactions.py

Given the reactions file for a `fix bond/react` reaction set
(the `molecule ...` / `react ...` lines, e.g. mol/in.reactions) and a
target LAMMPS data file, work out which of those molecule templates and
reactions could ever actually be used for that data file, and write out
a new, pruned reactions file containing only those.

Unlike an earlier version of this script, this does NOT need the
original combinatorial reaction-tree definition file, a --reaction-name,
a --mol-dir, or --max-bonds: every piece of information needed is
recovered directly from the reactions file's own `molecule`/
`react` lines and the *-pre/-post.molecule_template files they point to.

How it works
------------
Every `<label>_pre` molecule template (as produced by
data2reaction.py/react_molecule.py) is, by construction, exactly two
disconnected pieces in one file: the "already grown" reactant and a
fresh copy of the other reactant, with no bond yet between them. This
script parses each pre-template, splits it into its (exactly two)
connected components via its own Bonds section, and treats each
component as one reactant fragment -- with no need to know which
original base Reactants or earlier reactions it came from.

A fragment is "available" for the target data file if it is found, as
a subgraph match (atom typelabels and the fragment's own bonds; only
the fragment's own EdgeID atoms are allowed extra bonds to atoms
*outside* the fragment, representing growth/attachments its own
extraction deliberately trimmed away -- every other atom's real bond
count must exactly match its count within the fragment, since the
fragment asserts that is its complete bonding), either:
  - directly in the target data file, or
  - in the `<label>_post` molecule template of some OTHER reaction that
    is itself determined to be "needed" (since if that reaction can
    fire, it will eventually produce that exact structure).

A fragment with no known EdgeIDs at all (nothing was trimmed off it,
e.g. an isolated base reactant) therefore requires a match to a
*whole*, self-contained fragment (no atom in the match may have any
extra bond), since such a fragment is not expected to be embedded in
anything larger. But a fragment that *does* have EdgeIDs also still
requires every one of its non-EdgeID atoms -- e.g. an interior atom
whose own arm was truncated by the reaction generator's --max-bonds
cutoff, or a reacting center recorded with only the one bond this
fragment describes -- to match exactly too; only its designated EdgeID
atoms may have additional, unrelated real-world bonds.

A reaction is "needed" if both of its pre-template's two fragments are
available. Because availability can depend on other reactions' being
needed (cascading products), this is computed by iterating to a fixed
point: repeat a pass over all reactions, updating "needed", until no
label's status changes.

Usage:
    python3 prune_reactions.py \
        --reactions mol/in.reactions \
        --data-file final.data.tooshort \
        --output mol/in.reactions.pruned
"""

import argparse

import data2molecule as d2m
import data2reaction as d2r


def connected_components(counts, sections):
    """Split a parsed (counts, sections) fragment into its connected
    components (by its own Bonds section), returning a list of
    (sub_counts, sub_sections, edge_ids) tuples, one per component, each
    renumbered to start at atom id 1 (via d2m.renumber_atoms), and each
    carrying its share of `sections["_known_edge_ids"]` translated into
    its own new numbering.

    A component with no bonds at all (a single, unbonded atom) is
    treated as its own one-atom component.
    """
    atoms = sections.get("Atoms", [])
    atom_ids = [int(row[0]) for row in atoms]
    adjacency = {aid: set() for aid in atom_ids}
    for row in sections.get("Bonds", []):
        a1, a2 = int(row[2]), int(row[3])
        adjacency[a1].add(a2)
        adjacency[a2].add(a1)

    known_edge_ids = sections.get("_known_edge_ids", set())

    seen = set()
    component_id_sets = []
    for start in atom_ids:
        if start in seen:
            continue
        comp_ids = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            for neighbor in adjacency[current]:
                if neighbor not in comp_ids:
                    comp_ids.add(neighbor)
                    queue.append(neighbor)
        seen |= comp_ids
        component_id_sets.append(comp_ids)

    result = []
    for comp_ids in component_id_sets:
        comp_atoms = [row for row in atoms if int(row[0]) in comp_ids]
        comp_sections = {"Atoms": comp_atoms}
        for kind in ("Bonds", "Angles", "Dihedrals", "Impropers"):
            rows = sections.get(kind, [])
            comp_sections[kind] = [row for row in rows if int(row[2]) in comp_ids]
        for label_section in (
            "Atom Type Labels", "Bond Type Labels", "Angle Type Labels",
            "Dihedral Type Labels", "Improper Type Labels",
        ):
            if label_section in sections:
                comp_sections[label_section] = sections[label_section]

        comp_counts = {
            "atoms": len(comp_atoms),
            "atom types": counts.get("atom types", 0),
        }
        for kind, key in (
            ("Bonds", "bond"), ("Angles", "angle"),
            ("Dihedrals", "dihedral"), ("Impropers", "improper"),
        ):
            comp_counts[kind.lower()] = len(comp_sections[kind])
            comp_counts[f"{key} types"] = counts.get(f"{key} types", 0)

        comp_sections["_known_edge_ids"] = known_edge_ids & comp_ids
        new_counts, new_sections = d2m.renumber_atoms(comp_counts, comp_sections)
        result.append((new_counts, new_sections, new_sections["_known_edge_ids"]))

    return result


def fragment_present(
    frag_counts, frag_sections, edge_ids, full_type, full_adjacency, match_cache,
):
    """True if the fragment (frag_counts, frag_sections) is found as a
    subgraph match (atom typelabels + the fragment's own bonds) rooted
    at any of its own atoms, anywhere in the target graph (`full_type`/
    `full_adjacency`, as returned by
    data2reaction.build_atom_type_and_adjacency).

    Only the fragment's own EdgeID atoms (`edge_ids`) may have extra
    bonds to atoms outside the fragment in the match, representing
    growth/attachments the fragment's own extraction deliberately
    trimmed away; every other atom's real bond count must exactly equal
    its count within the fragment, since the fragment asserts that is
    its complete bonding (e.g. a reacting center whose only recorded
    bond is the one arm this fragment describes must not spuriously
    match a real, richly-coordinated atom just because it happens to
    have other, unrelated arms too).

    Equivalent to calling fragment_match_root() and checking whether it
    returned anything other than None; kept as a thin wrapper for
    readability at call sites that only care about presence/absence.
    """
    return fragment_match_root(
        frag_counts, frag_sections, edge_ids, full_type, full_adjacency, match_cache,
    ) is not None


def fragment_match_root(
    frag_counts, frag_sections, edge_ids, full_type, full_adjacency, match_cache,
):
    """Like fragment_present, but returns the matched atom id (in
    `full_type`/`full_adjacency`'s own numbering) that the fragment's
    own first-BFS-order atom mapped to, for one arbitrary matching
    embedding, or None if no match was found. Useful for reporting
    *where* a fragment was found (e.g. which real data-file atom, or
    which atom of another reaction's post-template)."""
    pattern_type, pattern_adjacency = d2r.build_atom_type_and_adjacency(
        frag_counts, frag_sections
    )
    if not pattern_type:
        return None

    cache_key = (
        frag_counts.get("atoms", 0), frag_counts.get("bonds", 0),
        tuple(sorted(pattern_type.items())),
        tuple(sorted((a, tuple(sorted(n))) for a, n in pattern_adjacency.items())),
        bool(edge_ids),
        id(full_type),
    )
    if cache_key in match_cache:
        return match_cache[cache_key]

    # Root the search at an arbitrary atom of the pattern (any atom
    # works; find_fragment_matches enumerates candidates by matching
    # type label first).
    center_id = next(iter(pattern_type))
    matches = d2r.find_fragment_matches(
        center_id, pattern_type, pattern_adjacency, full_type, full_adjacency
    )

    # find_fragment_matches allows every matched atom to have extra
    # bonds outside the fragment (representing other, unrelated arms).
    # That's correct for the fragment's own EdgeID atoms (where growth
    # is expected to continue), but WRONG for atoms that are not
    # EdgeIDs: those atoms' bonding is fully described by the fragment,
    # so a real match must have exactly that many bonds there too --
    # otherwise, e.g., a template's own center atom with only one bond
    # recorded (because the rest of the fragment was trimmed away by
    # --max-bonds) would spuriously match a real, richly-coordinated
    # atom just because it has extra, unrelated arms. So re-derive the
    # actual atom-by-atom mapping for each candidate and additionally
    # require every non-EdgeID atom's real degree to equal its fragment
    # degree exactly (EdgeID atoms may still have extra bonds).
    order = d2r._bfs_order(pattern_adjacency, center_id)
    found_root = None
    for root in matches:
        mapping = {order[0]: root}
        used = {root}

        def extend(idx):
            if idx == len(order):
                return True
            p = order[idx]
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

        if extend(1) and all(
            p in edge_ids
            or len(full_adjacency.get(mapping[p], ())) == len(pattern_adjacency[p])
            for p in order
        ):
            found_root = root
            break

    match_cache[cache_key] = found_root
    return found_root


def parse_reactions(path):
    """Parse a reactions file (molecule/react lines, e.g.
    mol/in.reactions) into:
      - preamble: every non-`molecule`/non-`react` line before the
        first `react` line (e.g. blank lines and the `fix ... bond/react
        ...` header line(s)), verbatim, in order.
      - molecule_lines: {label: [line, line]} (pre then post `molecule`
        lines) for every reaction label with `molecule` lines present.
      - molecule_paths: {label: [pre_path, post_path]} file paths (as
        written in the `molecule` lines, so relative to this script's
        own cwd, matching how the file itself refers to them).
      - react_lines: {label: line} (without any trailing ` &` line-
        continuation) for every reaction label with a `react` line.
      - react_order: reaction labels in the order their `react` lines
        appeared.
    """
    preamble = []
    molecule_lines = {}
    molecule_paths = {}
    react_lines = {}
    react_order = []
    seen_first_react = False

    with open(path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            tokens = stripped.split()

            if tokens and tokens[0] == "molecule" and not seen_first_react:
                # molecule NAME_pre/NAME_post file
                name = tokens[1]
                filepath = tokens[2]
                if name.endswith("_pre"):
                    label = name[: -len("_pre")]
                    molecule_lines.setdefault(label, []).append(line)
                    molecule_paths.setdefault(label, [None, None])[0] = filepath
                elif name.endswith("_post"):
                    label = name[: -len("_post")]
                    molecule_lines.setdefault(label, []).append(line)
                    molecule_paths.setdefault(label, [None, None])[1] = filepath
                else:
                    label = name
                    molecule_lines.setdefault(label, []).append(line)
                continue

            if tokens and tokens[0] == "react":
                seen_first_react = True
                label = tokens[1]
                # Strip a trailing line-continuation "&" (with any
                # surrounding whitespace); it's re-added on output only
                # for lines that aren't last.
                clean = stripped
                if clean.endswith("&"):
                    clean = clean[:-1].rstrip()
                react_lines[label] = clean
                react_order.append(label)
                continue

            if not seen_first_react:
                preamble.append(line)

    return preamble, molecule_lines, molecule_paths, react_lines, react_order


def write_pruned_reactions(path, preamble, molecule_lines, react_lines, kept_labels):
    with open(path, "w") as f:
        for label in kept_labels:
            for line in molecule_lines.get(label, []):
                f.write(line + "\n")
        for line in preamble:
            f.write(line + "\n")
        kept_react = [react_lines[label] for label in kept_labels if label in react_lines]
        for i, line in enumerate(kept_react):
            is_last = i == len(kept_react) - 1
            f.write("  " + line.strip() + ("" if is_last else " &") + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune a full combinatorial fix bond/react reaction "
        "set down to just the molecule templates/reactions that could "
        "ever actually be used (directly or as cascading reaction "
        "products) for one particular LAMMPS data file."
    )
    parser.add_argument(
        "--reactions", dest="reactions", required=True, metavar="FILE",
        help="The full combinatorial `molecule`/`react` command file to "
        "prune (e.g. mol/in.reactions). Its own `molecule` lines are "
        "the only source of pre/post molecule template paths used.",
    )
    parser.add_argument(
        "--data-file", dest="data_file", required=True, metavar="FILE",
        help="Target LAMMPS data file to read and check for molecule "
        "templates/reactions.",
    )
    parser.add_argument(
        "--output", dest="output", required=True, metavar="FILE",
        help="Where to write the pruned `molecule`/`react` command file.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each needed reaction's reactant fragment origins "
        "(data-file atom or producing reaction).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    prune_reactions(args.reactions, args.data_file, args.output, args.verbose)

def prune_reactions(reactions, data_file, output, verbose):
    preamble, molecule_lines, molecule_paths, react_lines, react_order = (
        parse_reactions(reactions)
    )

    target_counts, target_sections, extra_lines = d2m.parse_data_file(data_file)
    full_type, full_adjacency = d2r.build_atom_type_and_adjacency(
        target_counts, target_sections
    )
    match_cache = {}

    def match_root_in(counts, sections, edge_ids, other_full_type, other_full_adjacency):
        return fragment_match_root(
            counts, sections, edge_ids, other_full_type, other_full_adjacency,
            match_cache,
        )

    # Parse every label's pre-template into its two reactant fragments,
    # and every label's post-template up front.
    pre_fragments = {}  # label -> [(counts, sections, edge_ids), ...] (len 2)
    post_full = {}  # label -> (post_type, post_adjacency)
    for label, (pre_path, post_path) in molecule_paths.items():
        if pre_path is None or post_path is None:
            continue
        pre_counts, pre_sections = d2m.parse_molecule_template(pre_path)
        comps = connected_components(pre_counts, pre_sections)
        if len(comps) != 2:
            print(
                f"Warning: {pre_path} split into {len(comps)} connected "
                f"component(s) (expected 2); skipping reaction {label!r}."
            )
            continue
        pre_fragments[label] = comps

        post_counts, post_sections = d2m.parse_molecule_template(post_path)
        post_full[label] = d2r.build_atom_type_and_adjacency(post_counts, post_sections)

    # Availability against the *target data file* never changes across
    # iterations, so compute it once per label/fragment up front. Also
    # record, for --verbose reporting, the ("data", atom_id) origin of
    # each fragment that matches directly.
    direct_root = {
        label: [
            match_root_in(counts, sections, edge_ids, full_type, full_adjacency)
            for counts, sections, edge_ids in comps
        ]
        for label, comps in pre_fragments.items()
    }
    direct_available = {
        label: [root is not None for root in roots]
        for label, roots in direct_root.items()
    }
    # origin[label] = [origin_a, origin_b], each either
    # ("data", atom_id), ("reaction", other_label, atom_id), or None
    # (not yet/never found available).
    origin = {
        label: [
            ("data", direct_root[label][i]) if direct_available[label][i] else None
            for i in range(2)
        ]
        for label in pre_fragments
    }

    # Fixed-point iteration: a fragment is available if it matches the
    # data file directly, OR matches some other label's post-template
    # that is itself needed (i.e. reachable). Repeat until no label's
    # "needed" status changes.
    needed = {label: False for label in pre_fragments}
    changed = True
    while changed:
        changed = False
        for label, comps in pre_fragments.items():
            avail = []
            for i, (counts, sections, edge_ids) in enumerate(comps):
                if direct_available[label][i]:
                    avail.append(True)
                    continue
                found = False
                for other_label, post_type_adj in post_full.items():
                    if other_label == label or not needed[other_label]:
                        continue
                    other_full_type, other_full_adjacency = post_type_adj
                    root = match_root_in(counts, sections, edge_ids,
                                          other_full_type, other_full_adjacency)
                    if root is not None:
                        found = True
                        origin[label][i] = ("reaction", other_label, root)
                        break
                avail.append(found)
            new_needed = all(avail)
            if new_needed != needed[label]:
                needed[label] = new_needed
                changed = True

    def describe_origin(o):
        if o is None:
            return "NOT FOUND"
        if o[0] == "data":
            return f"data-file atom {o[1]}"
        return f"product of reaction {o[1]!r} (atom {o[2]})"

    if verbose:
        print()
        print("Reactant fragment origins:")
        for label in react_order:
            if label not in pre_fragments:
                continue
            o_a, o_b = origin[label]
            print(
                f"  {label}: reactant A <- {describe_origin(o_a)}; "
                f"reactant B <- {describe_origin(o_b)}"
            )

    kept_labels_set = {label for label, is_needed in needed.items() if is_needed}
    ordered_kept_labels = [label for label in react_order if label in kept_labels_set]

    write_pruned_reactions(
        output, preamble, molecule_lines, react_lines, ordered_kept_labels
    )

    print(
        f"{len(ordered_kept_labels)} of {len(pre_fragments)} reaction(s) "
        f"with their own molecule templates are needed for {data_file}."
    )
    print(f"Wrote pruned reactions file to {output}")


if __name__ == "__main__":
    main()
