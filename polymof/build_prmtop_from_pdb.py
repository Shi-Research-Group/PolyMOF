#!/usr/bin/env python3
"""
build_prmtop_from_pdb.py

Build a brand-new AMBER .prmtop file for a molecule described only by a
.pdb file (atomic elements, coordinates and CONECT bond records -- no
AMBER atom types, charges or force-field parameters), using one or more
existing, fully-parameterized .prmtop files as a parameter library.

This matches an atom by the local bonded-fragment signature it sits in ("same
neighborhood out to --max-bonds bonds") -- from *charges only* to *every*
per-atom and per-term quantity needed for a complete prmtop: AMBER atom type,
mass, charge, GB radius/screen/tree-chain/join/irotat, Lennard-Jones class, and
the full bonded topology (bonds, angles, proper dihedrals and GAFF-style
improper dihedrals) together with their force constants.

Because the new .pdb file has no atom types, the fragment signature
used to find an atom's equivalent in the library is built from
ATOMIC_NUMBER (element) instead of AMBER atom type. This is
necessarily coarser than the type-based matching used for charges
alone: two chemically distinct but locally isomorphic tautomer-like
environments (e.g. a "cc/nd" vs. an equivalent "nd/cc"-type
assignment around an aromatic N=C-N pattern) cannot be told apart by
element + topology alone, so a handful of such atoms may come out
with their type/charge role swapped relative to a specific "nominal"
answer. This is expected and harmless: both labelings are physically
equivalent parameterizations of the same local environment.

Bonds are taken directly from the new molecule's own topology (its
PDB CONECT records) -- these are not guessed. Angles and dihedrals
(proper and improper) are then *derived* from that bond graph plus
each atom's assigned AMBER atom type, and parameterized by looking up
each term's (canonicalized) atom-type tuple in a library built from
every bond/angle/dihedral term found in any of the old .prmtop files,
exactly the way AMBER force fields work (bonded parameters are fully
determined by atom type, never by global atom index).

Usage:
    python3 build_prmtop_from_pdb.py old1.prmtop old2.prmtop ... new.pdb
    python3 build_prmtop_from_pdb.py --max-bonds 4 old*.prmtop new.pdb
"""

import sys
import re
import argparse
import itertools
from collections import deque, defaultdict, Counter

FORMAT_RE = re.compile(r"^%FORMAT\(\s*(\d+)([a-zA-Z])(\d+)(?:\.(\d+))?\s*\)\s*$")

# Enough of the periodic table for organic/MOF force fields (extend as needed).
ELEMENT_TO_Z = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16,
    "CL": 17, "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22, "V": 23,
    "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
    "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37,
    "SR": 38, "Y": 39, "ZR": 40, "MO": 42, "RU": 44, "RH": 45, "PD": 46,
    "AG": 47, "CD": 48, "IN": 49, "SN": 50, "SB": 51, "I": 53, "CS": 55,
    "BA": 56, "PT": 78, "AU": 79, "HG": 80, "PB": 82,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a complete AMBER .prmtop for a new molecule "
        "described by a .pdb file (elements + coordinates + CONECT bonds "
        "only), using one or more fully-parameterized .prmtop files as a "
        "parameter library. Atoms are matched by local bonded-fragment "
        "signature (element-based, since the .pdb has no atom types)."
    )
    parser.add_argument(
        "--old-prmtops", nargs="+", metavar="old_prmtop", required=True,
        help="One or more prmtop file(s) with known atom types, charges "
        "and bonded parameters, used as the parameter library.",
    )
    parser.add_argument("--new-pdbs", nargs="+", required=True,
        help="PDB file (with CONECT records) describing the new molecule.")
    parser.add_argument(
        "--max-bonds", dest="max_bonds", type=int, default=3, metavar="N",
        help="Fragment radius, in bonds, used to determine whether an atom "
        "in the new molecule is equivalent to one in the library. Atoms "
        "with no match at exactly this radius cause the script to abort "
        "(re-run with a different --max-bonds if that happens). Default: 3.",
    )
    parser.add_argument(
        "--no-neutralize", dest="neutralize", action="store_false",
        help="Skip net-charge correction (shifting every atom's charge by "
        "an equal small amount so the output's net charge matches the "
        "library's average net charge, rounded to the nearest integer).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat any remaining ambiguity as a hard error instead of a "
        "warning: abort if any ambiguous atom could not be resolved via "
        "the bonded ring-partner (group-consistent) method and had to "
        "fall back to a per-atom majority vote, or if any bond/angle/"
        "dihedral parameter could not be found in the library and a "
        "generic fallback default had to be used. Use a larger "
        "--max-bonds to resolve most such cases instead of relaxing "
        "--strict.",
    )
    args = parser.parse_args()
    if args.max_bonds < 0:
        parser.error("--max-bonds must be >= 0")
    return args


# --------------------------------------------------------------------------
# Reading existing (library) prmtop files
# --------------------------------------------------------------------------

class PrmtopFile:
    """Read-only parser for an existing AMBER prmtop file: indexes every
    %FLAG section (name -> (data_start, data_end, (count_per_line, ftype,
    width, decimals))) so its values can be pulled out with get_values()."""

    def __init__(self, path):
        with open(path) as f:
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
                    raise ValueError(f"Expected %FORMAT(...) after %FLAG {name}, got: {fmt_line!r}")
                count_per_line = int(m.group(1))
                ftype = m.group(2).upper()
                width = int(m.group(3))
                decimals = int(m.group(4)) if m.group(4) else 0
                data_start = fmt_idx + 1
                j = data_start
                while j < n and not self.lines[j].startswith("%"):
                    j += 1
                self.sections[name] = (data_start, j, (count_per_line, ftype, width, decimals))
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
        if name not in self.sections:
            return []
        data_start, data_end, (count_per_line, ftype, width, decimals) = self.sections[name]
        values = []
        for field in self._iter_fields(data_start, data_end, width):
            text = field.strip()
            if text == "":
                continue
            if ftype == "I":
                values.append(int(text))
            elif ftype in ("E", "F", "D"):
                values.append(float(text.replace("D", "E").replace("d", "e")))
            else:
                values.append(text)
        return values


def build_adjacency(prm):
    """{atom_id (1-based): set(bonded atom_ids)} from an already-parsed
    PrmtopFile's own BONDS_INC_HYDROGEN / BONDS_WITHOUT_HYDROGEN."""
    n_atoms = len(prm.get_values("AMBER_ATOM_TYPE"))
    adjacency = {aid: set() for aid in range(1, n_atoms + 1)}
    for section in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = prm.get_values(section)
        for k in range(0, len(values), 3):
            ib, jb = values[k], values[k + 1]
            a1 = ib // 3 + 1
            a2 = jb // 3 + 1
            adjacency[a1].add(a2)
            adjacency[a2].add(a1)
    return adjacency


def fragment_signature(adjacency, label, root, max_bonds):
    """Canonical (renumbering-invariant) signature of the fragment
    centered on `root` out to `max_bonds` bonds, using 1-D
    Weisfeiler-Leman color refinement over `label` (element/atomic
    number here, rather than AMBER atom type, since the new molecule's
    .pdb has no atom types to match on)."""
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
    edge_ids = {aid for aid in keep_ids if any(nb not in keep_ids for nb in adjacency[aid])}
    sub_adjacency = {aid: (adjacency[aid] & keep_ids) for aid in keep_ids}

    color = {aid: (label[aid], aid in edge_ids, aid == root) for aid in keep_ids}
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

    n_bonds = sum(len(nbs) for nbs in sub_adjacency.values()) // 2
    return (len(keep_ids), n_bonds, color[root], tuple(sorted(color.values())))


# --------------------------------------------------------------------------
# Parsing the new molecule's .pdb (elements + coordinates + CONECT bonds)
# --------------------------------------------------------------------------

def parse_pdb(path):
    """Return (atoms, bonds): atoms is {serial: {"name","element","resname",
    "resnum"}}; bonds is a set of (min(a,b), max(a,b)) tuples from CONECT."""
    atoms = {}
    bonds = set()
    with open(path) as f:
        for line in f:
            rec = line[:6]
            if rec in ("ATOM  ", "HETATM"):
                serial = int(line[6:11])
                name = line[12:16].strip()
                resname = line[17:20].strip()
                resnum = line[22:26].strip()
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if elem:
                    elem = elem.upper()
                else:
                    m = re.match(r"[A-Za-z]+", name)
                    elem = m.group().upper() if m else "X"
                    # names like "C1"/"H2" -> strip trailing digits already
                    # handled by the regex; two-letter elements in atom
                    # names (e.g. "CL1") are rare here but keep as-is.
                if elem not in ELEMENT_TO_Z and len(elem) > 1:
                    elem = elem[0]
                atoms[serial] = {"name": name, "element": elem, "resname": resname, "resnum": resnum}
            elif rec == "CONECT":
                parts = line.split()[1:]
                if not parts:
                    continue
                nums = [int(p) for p in parts]
                a = nums[0]
                for b in nums[1:]:
                    bonds.add((min(a, b), max(a, b)))
    return atoms, bonds


# --------------------------------------------------------------------------
# Parameter library built from the old prmtop file(s)
# --------------------------------------------------------------------------

def _close(a, b, tol=1e-4):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _add_dihedral_term(store, key, term, path, kind, atom_types):
    """Add one (k, period, phase, scee, scnb) Fourier term for a bonded
    dihedral/improper type-key, raising a hard error if it genuinely
    conflicts with a term already recorded for the same key and the same
    periodicity (a real force-field disagreement) rather than simply
    contributing an additional, distinct Fourier component (a normal
    multi-term torsion, which every source file is expected to agree on
    identically term-for-term).
    """
    existing_terms = store[key]
    period = term[1]
    for other in existing_terms:
        if other[1] == period and not _close(other, term, tol=1e-3):
            sys.exit(
                f"ERROR: conflicting {kind} parameters for atom types {atom_types} "
                f"(type-key {key}), periodicity {period}: {other} (from an earlier "
                f"file) vs {term} (from {path}). The library files do not agree on "
                f"a unique force field term for this periodicity; refusing to guess "
                f"which is correct."
            )
    existing_terms.add(term)


class Library:
    def __init__(self, prmtop_paths, max_bonds):
        self.max_bonds = max_bonds
        self.sig_to_atoms = defaultdict(list)
        self.bond_params = {}
        self.angle_params = {}
        self.dihedral_params = defaultdict(set)
        self.improper_params = defaultdict(set)
        self.type_self_lj = {}
        self.net_charges = []
        self.radius_set = None
        self.file_adjacency = {}
        for path in prmtop_paths:
            self._ingest(path)

    def _ingest(self, path):
        prm = PrmtopFile(path)
        types = prm.get_values("AMBER_ATOM_TYPE")
        n_atoms = len(types)
        if n_atoms == 0:
            return
        atomic_numbers = prm.get_values("ATOMIC_NUMBER")
        charges = prm.get_values("CHARGE")
        masses = prm.get_values("MASS")
        radii = prm.get_values("RADII")
        screen = prm.get_values("SCREEN")
        tree = prm.get_values("TREE_CHAIN_CLASSIFICATION")
        join = prm.get_values("JOIN_ARRAY")
        irotat = prm.get_values("IROTAT")
        radius_set_vals = prm.get_values("RADIUS_SET")
        if radius_set_vals and self.radius_set is None:
            self.radius_set = radius_set_vals[0]

        adjacency = build_adjacency(prm)
        label = {i + 1: atomic_numbers[i] for i in range(n_atoms)}
        self.net_charges.append(sum(charges))
        self.file_adjacency[path] = adjacency

        for aid in range(1, n_atoms + 1):
            sig = fragment_signature(adjacency, label, aid, self.max_bonds)
            self.sig_to_atoms[sig].append({
                "type": types[aid - 1],
                "charge": charges[aid - 1],
                "mass": masses[aid - 1],
                "radius": radii[aid - 1] if radii else 0.0,
                "screen": screen[aid - 1] if screen else 0.0,
                "tree": tree[aid - 1] if tree else "M",
                "join": join[aid - 1] if join else 0,
                "irotat": irotat[aid - 1] if irotat else 0,
                "atomic_number": atomic_numbers[aid - 1],
                "source": path,
                "atom_id": aid,
            })

        kb_arr = prm.get_values("BOND_FORCE_CONSTANT")
        req_arr = prm.get_values("BOND_EQUIL_VALUE")
        for section in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
            values = prm.get_values(section)
            for k in range(0, len(values), 3):
                ib, jb, icb = values[k], values[k + 1], values[k + 2]
                a1, a2 = ib // 3 + 1, jb // 3 + 1
                key = tuple(sorted((types[a1 - 1], types[a2 - 1])))
                val = (round(kb_arr[icb - 1], 6), round(req_arr[icb - 1], 6))
                existing = self.bond_params.get(key)
                if existing is not None and not _close(existing, val):
                    sys.exit(
                        f"ERROR: conflicting BOND parameters for atom types {key}: "
                        f"{existing} (from an earlier file) vs {val} (from {path}). "
                        f"The library files do not agree on a unique force field for this "
                        f"type pair; refusing to guess which is correct."
                    )
                self.bond_params.setdefault(key, val)

        kt_arr = prm.get_values("ANGLE_FORCE_CONSTANT")
        teq_arr = prm.get_values("ANGLE_EQUIL_VALUE")
        for section in ("ANGLES_INC_HYDROGEN", "ANGLES_WITHOUT_HYDROGEN"):
            values = prm.get_values(section)
            for k in range(0, len(values), 4):
                ia, ja, ka, ict = values[k], values[k + 1], values[k + 2], values[k + 3]
                a1, a2, a3 = ia // 3 + 1, ja // 3 + 1, ka // 3 + 1
                t1, t2, t3 = types[a1 - 1], types[a2 - 1], types[a3 - 1]
                ends = tuple(sorted((t1, t3)))
                key = (ends[0], t2, ends[1])
                val = (round(kt_arr[ict - 1], 6), round(teq_arr[ict - 1], 6))
                existing = self.angle_params.get(key)
                if existing is not None and not _close(existing, val):
                    sys.exit(
                        f"ERROR: conflicting ANGLE parameters for atom types {key}: "
                        f"{existing} (from an earlier file) vs {val} (from {path}). "
                        f"The library files do not agree on a unique force field for this "
                        f"type triple; refusing to guess which is correct."
                    )
                self.angle_params.setdefault(key, val)

        kp_arr = prm.get_values("DIHEDRAL_FORCE_CONSTANT")
        per_arr = prm.get_values("DIHEDRAL_PERIODICITY")
        phase_arr = prm.get_values("DIHEDRAL_PHASE")
        scee_arr = prm.get_values("SCEE_SCALE_FACTOR")
        scnb_arr = prm.get_values("SCNB_SCALE_FACTOR")
        for section in ("DIHEDRALS_INC_HYDROGEN", "DIHEDRALS_WITHOUT_HYDROGEN"):
            values = prm.get_values(section)
            for k in range(0, len(values), 5):
                ia, ja, ka, la, icp = values[k:k + 5]
                is_improper = la < 0
                a1 = ia // 3 + 1
                a2 = ja // 3 + 1
                a3 = abs(ka) // 3 + 1
                a4 = abs(la) // 3 + 1
                t1, t2, t3, t4 = types[a1 - 1], types[a2 - 1], types[a3 - 1], types[a4 - 1]
                term = (
                    round(kp_arr[icp - 1], 6),
                    round(per_arr[icp - 1], 1),
                    round(phase_arr[icp - 1], 6),
                    round(scee_arr[icp - 1], 4) if icp - 1 < len(scee_arr) else 1.2,
                    round(scnb_arr[icp - 1], 4) if icp - 1 < len(scnb_arr) else 2.0,
                )
                if is_improper:
                    key = (t3, tuple(sorted((t1, t2, t4))))
                    _add_dihedral_term(self.improper_params, key, term, path, "IMPROPER", (t1, t2, t3, t4))
                else:
                    key = min((t1, t2, t3, t4), (t4, t3, t2, t1))
                    _add_dihedral_term(self.dihedral_params, key, term, path, "DIHEDRAL", (t1, t2, t3, t4))


        ntypes = prm.get_values("POINTERS")[1]
        aty = prm.get_values("ATOM_TYPE_INDEX")
        ico = prm.get_values("NONBONDED_PARM_INDEX")
        acoef = prm.get_values("LENNARD_JONES_ACOEF")
        bcoef = prm.get_values("LENNARD_JONES_BCOEF")
        for aid in range(1, n_atoms + 1):
            t = types[aid - 1]
            if t in self.type_self_lj:
                continue
            c = aty[aid - 1]
            idx = ico[ntypes * (c - 1) + (c - 1)]
            if idx > 0:
                self.type_self_lj[t] = (acoef[idx - 1], bcoef[idx - 1])


# --------------------------------------------------------------------------
# Matching new-molecule atoms to library atoms by element-based fragment
# --------------------------------------------------------------------------

def find_raw_matches(new_adjacency, new_label, library, max_bonds):
    """First pass: for every new-molecule atom, find its raw list of
    equivalent library atoms (same element-based fragment signature) at
    the requested search radius (max_bonds). Doesn't resolve any type
    ambiguity yet -- see resolve_types() for that.

    If any atom has no library match at this exact radius, the caller is
    expected to abort the run (see main()) rather than silently retrying
    at a smaller radius -- the user can re-invoke with a smaller
    --max-bonds if that is what they want."""
    raw_matches = {}
    unmatched = []
    for aid in sorted(new_label):
        sig = fragment_signature(new_adjacency, new_label, aid, max_bonds)
        matches = library.sig_to_atoms.get(sig)
        if matches is None:
            unmatched.append(aid)
            continue
        raw_matches[aid] = {"matches": matches, "radius": max_bonds}
    return raw_matches, unmatched


def resolve_types(new_adjacency, raw_matches, library):
    """Resolve each atom's final AMBER atom type from its raw match list.

    Most atoms are unambiguous (every matching library atom has the same
    type) and are simply assigned that type. Atoms whose matches disagree
    on type ("ambiguous") typically arise from GAFF's element-invisible
    aromatic-ring tautomer bookkeeping: a "cc"/"cd" hub carbon must always
    be paired with a "nd"/"nc" ring-nitrogen partner respectively (GAFF
    only ever produces the combinations "cc+nd" or "cd+nc", never
    "cc+nc" or "cd+nd"), so resolving each ambiguous atom's type
    *independently* (e.g. by a per-atom majority vote) can accidentally
    produce an invalid/nonexistent combination if the hub and its partner
    happen to vote for opposite tautomers.

    To avoid that, directly-bonded ambiguous atoms are grouped together
    and resolved *as a unit*, against a single old-molecule source file:
    for each candidate file, we require not merely that every atom in the
    group has *some* matching atom in that file, but that there is an
    assignment of old atoms to group atoms whose *bonding among
    themselves* mirrors the new molecule's own bonding within the group
    (e.g. if the group's hub and ring-nitrogen are bonded to each other,
    their chosen old-molecule counterparts must be bonded to each other
    too). This guarantees a self-consistent combination -- copied
    verbatim from one real, chemically valid molecule -- rather than
    stitched together atom-by-atom from unrelated positions.
    """
    type_counts_per_atom = {}
    ambiguous_info = []
    for aid, info in raw_matches.items():
        counts = Counter(m["type"] for m in info["matches"])
        type_counts_per_atom[aid] = counts
        if len(counts) > 1:
            ambiguous_info.append((aid, dict(counts)))

    def majority_vote(aid):
        counts = type_counts_per_atom[aid]
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    ambiguous_atoms = {aid for aid, counts in type_counts_per_atom.items() if len(counts) > 1}

    # Group directly-bonded ambiguous atoms into connected components, so
    # e.g. a cc/cd hub and its nc/nd ring-nitrogen partner are resolved
    # together rather than independently.
    visited = set()
    groups = []
    for aid in sorted(ambiguous_atoms):
        if aid in visited:
            continue
        stack = [aid]
        comp = []
        visited.add(aid)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in new_adjacency[cur]:
                if nb in ambiguous_atoms and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        groups.append(sorted(comp))

    assigned_type = {}
    resolved_by_group = set()
    for group in groups:
        if len(group) == 1:
            continue  # no bonded ambiguous partner to coordinate with; fall through to majority vote below.

        internal_edges = [
            (u, v) for u in group for v in new_adjacency[u]
            if v in group and u < v
        ]

        candidates_by_file = defaultdict(lambda: defaultdict(list))
        for aid in group:
            for m in raw_matches[aid]["matches"]:
                candidates_by_file[m["source"]][aid].append(m)

        best = None  # (coverage, file, {aid: match})
        for f in sorted(candidates_by_file):
            per_atom = candidates_by_file[f]
            cand_lists = [per_atom.get(aid, []) for aid in group]
            if any(not c for c in cand_lists):
                continue  # this file doesn't have a match for every group atom
            adjacency_f = library.file_adjacency[f]
            found = None
            for combo in itertools.product(*cand_lists):
                old_id = {group[i]: combo[i]["atom_id"] for i in range(len(group))}
                if all(old_id[v] in adjacency_f[old_id[u]] for (u, v) in internal_edges):
                    found = {group[i]: combo[i] for i in range(len(group))}
                    break
            if found is not None:
                coverage = len(group)
                if best is None or coverage > best[0]:
                    best = (coverage, f, found)

        if best is not None:
            _, _, chosen = best
            for aid, m in chosen.items():
                assigned_type[aid] = m["type"]
                resolved_by_group.add(aid)
        else:
            # No single source file offers an internally-consistent
            # assignment for the whole group; fall back to each atom's
            # own majority vote.
            for aid in group:
                assigned_type[aid] = majority_vote(aid)

    for aid in type_counts_per_atom:
        if aid not in assigned_type:
            assigned_type[aid] = majority_vote(aid)

    majority_vote_atoms = [aid for aid, counts in type_counts_per_atom.items()
                            if len(counts) > 1 and aid not in resolved_by_group]

    return assigned_type, ambiguous_info, majority_vote_atoms


def match_atoms(new_adjacency, new_label, library, max_bonds):
    """Combine find_raw_matches() + resolve_types() into per-atom
    parameter records (type, charge, mass, radii, ...), matching the
    interface the rest of the script expects."""
    raw_matches, unmatched = find_raw_matches(new_adjacency, new_label, library, max_bonds)
    if unmatched:
        return {}, unmatched, [], []

    assigned_type, ambiguous, majority_vote_atoms = resolve_types(new_adjacency, raw_matches, library)

    assigned = {}
    for aid, info in raw_matches.items():
        matches = info["matches"]
        best_type = assigned_type[aid]
        charge = sum(m["charge"] for m in matches) / len(matches)
        rep = next((m for m in matches if m["type"] == best_type), matches[0])
        record = dict(rep)
        record["type"] = best_type
        record["charge"] = charge
        record["radius_used"] = info["radius"]
        assigned[aid] = record
    return assigned, unmatched, ambiguous, majority_vote_atoms


# --------------------------------------------------------------------------
# Building the new molecule's bonded topology
# --------------------------------------------------------------------------

def build_angles(new_adjacency):
    angles = []
    for j in sorted(new_adjacency):
        neighbors = sorted(new_adjacency[j])
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                angles.append((neighbors[a], j, neighbors[b]))
    return angles


def build_proper_dihedrals(new_bonds, new_adjacency):
    quartets = []
    for (a, b) in sorted(new_bonds):
        left = sorted(new_adjacency[a] - {b})
        right = sorted(new_adjacency[b] - {a})
        for i in left:
            for l in right:
                if i == l:
                    continue
                quartets.append((i, a, b, l))

    groups = defaultdict(list)
    for q in quartets:
        i, j, k, l = q
        groups[frozenset((i, l))].append(q)

    no14 = {}
    for pair, qs in groups.items():
        i, l = tuple(pair) if len(pair) == 2 else (next(iter(pair)),) * 2
        short_path = (l in new_adjacency[i]) or bool(new_adjacency[i] & new_adjacency[l])
        qs_sorted = sorted(qs)
        if short_path:
            for q in qs_sorted:
                no14[q] = True
        else:
            no14[qs_sorted[0]] = False
            for q in qs_sorted[1:]:
                no14[q] = True
    return quartets, no14


def build_impropers(new_adjacency, assigned_type, library):
    impropers = []
    for hub in sorted(new_adjacency):
        neighbors = sorted(new_adjacency[hub])
        if len(neighbors) != 3:
            continue
        hub_type = assigned_type[hub]
        ntypes3 = tuple(sorted(assigned_type[n] for n in neighbors))
        key = (hub_type, ntypes3)
        if key not in library.improper_params:
            continue
        ordered = sorted(neighbors, key=lambda n: (assigned_type[n], n))
        i, j, l = ordered
        impropers.append((i, j, hub, l, library.improper_params[key]))
    return impropers


def build_exclusions(n_atoms, new_bonds, angles, proper_quartets):
    excl = defaultdict(set)
    for (a, b) in new_bonds:
        lo, hi = min(a, b), max(a, b)
        excl[lo].add(hi)
    for (i, j, k) in angles:
        lo, hi = min(i, k), max(i, k)
        excl[lo].add(hi)
    for (i, j, k, l) in proper_quartets:
        lo, hi = min(i, l), max(i, l)
        excl[lo].add(hi)
    return excl


# --------------------------------------------------------------------------
# Writing the new prmtop file
# --------------------------------------------------------------------------

def format_int_field(value, width):
    return str(value).rjust(width)


def format_float_field(value, width, decimals):
    text = f"{value:.{decimals}E}"
    return text.rjust(width)


def format_str_field(value, width):
    return str(value).ljust(width)


def write_section(out, name, values, count_per_line, ftype, width, decimals):
    out.append(f"%FLAG {name}\n")
    if ftype in ("E", "F", "D"):
        fmt_letter = "E"
        out.append(f"%FORMAT({count_per_line}{fmt_letter}{width}.{decimals})\n")
    elif ftype == "I":
        out.append(f"%FORMAT({count_per_line}I{width})\n")
    else:
        out.append(f"%FORMAT({count_per_line}a{width})\n")
    for k in range(0, len(values), count_per_line):
        chunk = values[k:k + count_per_line]
        if ftype in ("E", "F", "D"):
            fields = [format_float_field(v, width, decimals) for v in chunk]
        elif ftype == "I":
            fields = [format_int_field(v, width) for v in chunk]
        else:
            fields = [format_str_field(v, width) for v in chunk]
        out.append("".join(fields) + "\n")
    if not values:
        out.append("\n")

def main():
    args = parse_args()
    build_prmtop(args.old_prmtops, args.new_pdbs, args.max_bonds, args.neutralize, args.strict)

def build_prmtop(old_prmtops, new_pdbs, max_bonds=10, neutralize=True, strict=True):

    print(f"Reading parameter library from {len(old_prmtops)} old prmtop file(s)...", end='', flush=True)
    library = Library(old_prmtops, max_bonds)
    print("  done", flush=True)

    for new_pdb in new_pdbs:
        build_prmtop_for_pdb(library, new_pdb, max_bonds, neutralize, strict)

def build_prmtop_for_pdb(library, new_pdb, max_bonds, neutralize, strict):
    output_prmtop = new_pdb.replace(".pdb", ".prmtop")
    if output_prmtop == new_pdb:
        print("PDB file must have a pdb extension")
        sys.exit(1)

    atoms, pdb_bonds = parse_pdb(new_pdb)
    serials = sorted(atoms)
    n_atoms = len(serials)
    if serials != list(range(1, n_atoms + 1)):
        sys.exit(
            "This script requires PDB atom serial numbers to be a contiguous "
            "1..N sequence (matching AMBER atom numbering); found gaps/reordering."
        )

    new_adjacency = {aid: set() for aid in range(1, n_atoms + 1)}
    for (a, b) in pdb_bonds:
        new_adjacency[a].add(b)
        new_adjacency[b].add(a)

    new_label = {}
    for aid in range(1, n_atoms + 1):
        elem = atoms[aid]["element"]
        z = ELEMENT_TO_Z.get(elem)
        if z is None:
            sys.exit(f"Unknown element {elem!r} for atom {aid} ({atoms[aid]['name']}).")
        new_label[aid] = z

    assigned, unmatched, ambiguous, majority_vote_atoms = match_atoms(new_adjacency, new_label, library, max_bonds)

    if unmatched:
        sys.exit(
            f"ERROR: {new_pdb} {len(unmatched)} atom(s) had no library match at --max-bonds {max_bonds}; "
            f"these cannot be parameterized: {unmatched}. "
            f"Re-run with a smaller --max-bonds value to allow a coarser (shorter-range) "
            f"fragment match, or expand the library to cover this local environment."
        )

    if ambiguous:
        n_group = len(ambiguous) - len(majority_vote_atoms)
        print(f"Note: {len(ambiguous)} atom(s) matched more than one AMBER atom type in the library. "
              f"{n_group} of these were resolved by finding a bonded ring-partner and picking a "
              "single internally-consistent source file for the whole group; ")
        if len(majority_vote_atoms) > 0:
            print(f"{len(majority_vote_atoms)} had no such bonded ambiguous partner (or no single file "
                  "covered the group consistently) and were resolved by per-atom majority vote instead. ")
            print("Majority vote atom(s): "
              + ", ".join(
                  f"{aid}({atoms[aid]['name']}):{counts}"
                  for aid, counts in ambiguous if aid in majority_vote_atoms))

    if strict and majority_vote_atoms:
        sys.exit(
            f"ERROR (--strict): {len(majority_vote_atoms)} atom(s) could not be resolved via the "
            f"bonded ring-partner (group-consistent) method and fell back to a per-atom majority "
            f"vote: {majority_vote_atoms}. Re-run with a larger --max-bonds (which often "
            f"disambiguates these directly) or without --strict."
        )

    assigned_type = {aid: assigned[aid]["type"] for aid in assigned}

    # ---- Lennard-Jones classes: group AMBER atom types actually used by
    # identical self (A,B) Lennard-Jones coefficients (this is exactly how
    # AMBER/tleap collapse e.g. ca/cc/cd/cp -- chemically distinct bonded
    # types that nonetheless share one vdW class -- into fewer NTYPES than
    # there are distinct AMBER_ATOM_TYPE labels).
    def lj_key(t):
        a, b = library.type_self_lj.get(t, (0.0, 0.0))
        return (round(a, 4), round(b, 4))

    type_to_class = {}
    class_key_list = []
    for aid in range(1, n_atoms + 1):
        t = assigned_type[aid]
        if t in type_to_class:
            continue
        key = lj_key(t)
        found = None
        for idx, k in enumerate(class_key_list):
            if k == key:
                found = idx + 1
                break
        if found is None:
            class_key_list.append(key)
            found = len(class_key_list)
        type_to_class[t] = found
    ntypes = len(class_key_list)

    import math

    def sigma_eps(a, b):
        if b == 0:
            return 0.0, 0.0
        sigma6 = a / b
        sigma = sigma6 ** (1 / 6)
        eps = b / (4 * sigma ** 6)
        return sigma, eps

    class_sigma_eps = [sigma_eps(*k) for k in class_key_list]

    n_pairs = ntypes * (ntypes + 1) // 2
    acoef = [0.0] * n_pairs
    bcoef = [0.0] * n_pairs
    ico = [0] * (ntypes * ntypes)
    for ci in range(1, ntypes + 1):
        for cj in range(1, ntypes + 1):
            i, j = sorted((ci, cj))
            idx = j * (j - 1) // 2 + i
            ico[(ci - 1) * ntypes + (cj - 1)] = idx
    for j in range(1, ntypes + 1):
        for i in range(1, j + 1):
            idx = j * (j - 1) // 2 + i
            si, ei = class_sigma_eps[i - 1]
            sj, ej = class_sigma_eps[j - 1]
            sij = (si + sj) / 2
            eij = math.sqrt(ei * ej)
            acoef[idx - 1] = 4 * eij * sij ** 12
            bcoef[idx - 1] = 4 * eij * sij ** 6

    # ---- Bonds (topology comes straight from the PDB; parameters from the
    # type-keyed library).
    bond_param_index = {}
    bond_force_constants = []
    bond_equil_values = []
    bonds_h, bonds_noh = [], []
    n_bond_warnings = 0
    bond_fallback_items = []
    for (a, b) in sorted(pdb_bonds):
        t1, t2 = assigned_type[a], assigned_type[b]
        key = tuple(sorted((t1, t2)))
        params = library.bond_params.get(key)
        if params is None:
            n_bond_warnings += 1
            bond_fallback_items.append((a, b, key))
            params = (300.0, 1.5)
        if key not in bond_param_index:
            bond_param_index[key] = len(bond_force_constants) + 1
            bond_force_constants.append(params[0])
            bond_equil_values.append(params[1])
        icb = bond_param_index[key]
        entry = (3 * (a - 1), 3 * (b - 1), icb)
        if new_label[a] == 1 or new_label[b] == 1:
            bonds_h.append(entry)
        else:
            bonds_noh.append(entry)

    # ---- Angles.
    angles = build_angles(new_adjacency)
    angle_param_index = {}
    angle_force_constants = []
    angle_equil_values = []
    angles_h, angles_noh = [], []
    n_angle_warnings = 0
    angle_fallback_items = []
    for (i, j, k) in angles:
        t1, t2, t3 = assigned_type[i], assigned_type[j], assigned_type[k]
        ends = tuple(sorted((t1, t3)))
        key = (ends[0], t2, ends[1])
        params = library.angle_params.get(key)
        if params is None:
            n_angle_warnings += 1
            angle_fallback_items.append((i, j, k, key))
            params = (50.0, 109.5 * math.pi / 180)
        if key not in angle_param_index:
            angle_param_index[key] = len(angle_force_constants) + 1
            angle_force_constants.append(params[0])
            angle_equil_values.append(params[1])
        ict = angle_param_index[key]
        entry = (3 * (i - 1), 3 * (j - 1), 3 * (k - 1), ict)
        if new_label[i] == 1 or new_label[j] == 1 or new_label[k] == 1:
            angles_h.append(entry)
        else:
            angles_noh.append(entry)

    # ---- Proper + improper dihedrals.
    proper_quartets, no14 = build_proper_dihedrals(pdb_bonds, new_adjacency)
    impropers = build_impropers(new_adjacency, assigned_type, library)

    dih_param_index = {}
    dih_force_constants = []
    dih_periodicities = []
    dih_phases = []
    dih_scee = []
    dih_scnb = []
    dih_h, dih_noh = [], []
    n_dih_warnings = 0
    dih_fallback_items = []

    def get_dih_index(term):
        if term not in dih_param_index:
            dih_param_index[term] = len(dih_force_constants) + 1
            dih_force_constants.append(term[0])
            dih_periodicities.append(term[1])
            dih_phases.append(term[2])
            dih_scee.append(term[3])
            dih_scnb.append(term[4])
        return dih_param_index[term]

    def emit_dihedral(i, j, k, l, terms, is_no14, is_improper):
        has_h = any(new_label[x] == 1 for x in (i, j, k, l))
        bucket = dih_h if has_h else dih_noh
        ia, ja = 3 * (i - 1), 3 * (j - 1)
        ka_mag, la_mag = 3 * (k - 1), 3 * (l - 1)
        # AMBER encodes "no 1-4"/"improper" flags as a negative sign on the
        # 3rd/4th atom index; that's impossible for the very first atom
        # (whose 0-based*3 value is 0), so swap the quartet's direction in
        # that rare case instead (same physical term, reversed order).
        if (is_no14 and ka_mag == 0) or (is_improper and la_mag == 0):
            i, j, k, l = l, k, j, i
            ia, ja = 3 * (i - 1), 3 * (j - 1)
            ka_mag, la_mag = 3 * (k - 1), 3 * (l - 1)
        ka = -ka_mag if (is_no14 and ka_mag != 0) else ka_mag
        la = -la_mag if (is_improper and la_mag != 0) else la_mag
        for term in terms:
            icp = get_dih_index(term)
            bucket.append((ia, ja, ka, la, icp))

    for q in proper_quartets:
        i, j, k, l = q
        t1, t2, t3, t4 = assigned_type[i], assigned_type[j], assigned_type[k], assigned_type[l]
        key = min((t1, t2, t3, t4), (t4, t3, t2, t1))
        terms = library.dihedral_params.get(key)
        if not terms:
            n_dih_warnings += 1
            dih_fallback_items.append((i, j, k, l, key))
            terms = {(0.0, 2.0, 0.0, 1.2, 2.0)}
        emit_dihedral(i, j, k, l, sorted(terms), no14[q], False)

    for (i, j, k, l, terms) in impropers:
        emit_dihedral(i, j, k, l, sorted(terms), False, True)

    if n_bond_warnings or n_angle_warnings or n_dih_warnings:
        print(f"WARNING: used fallback default parameters for {n_bond_warnings} bond(s), "
              f"{n_angle_warnings} angle(s), {n_dih_warnings} dihedral(s) not found in the library.")

    if strict and (n_bond_warnings or n_angle_warnings or n_dih_warnings):
        sys.exit(
            "ERROR (--strict): fallback default parameters were required for "
            f"{n_bond_warnings} bond(s), {n_angle_warnings} angle(s), {n_dih_warnings} "
            "dihedral(s) not found in the library:\n"
            + "\n".join(f"  bond {a}-{b} {key}" for (a, b, key) in bond_fallback_items)
            + ("\n" if bond_fallback_items else "")
            + "\n".join(f"  angle {i}-{j}-{k} {key}" for (i, j, k, key) in angle_fallback_items)
            + ("\n" if angle_fallback_items else "")
            + "\n".join(f"  dihedral {i}-{j}-{k}-{l} {key}" for (i, j, k, l, key) in dih_fallback_items)
            + "\nRe-run with a larger --max-bonds (which often disambiguates the "
            "underlying tautomer-labeling ambiguity directly) or without --strict."
        )

    # ---- Exclusions.
    excl_map = build_exclusions(n_atoms, pdb_bonds, angles, proper_quartets)
    number_excluded = []
    excluded_list = []
    for aid in range(1, n_atoms + 1):
        partners = sorted(excl_map.get(aid, ()))
        if partners:
            number_excluded.append(len(partners))
            excluded_list.extend(partners)
        else:
            number_excluded.append(1)
            excluded_list.append(0)

    # ---- Charges, with optional net-charge neutralization.
    charges = [assigned[aid]["charge"] for aid in range(1, n_atoms + 1)]
    if neutralize and library.net_charges:
        target = round(sum(library.net_charges) / len(library.net_charges))
        raw = sum(charges)
        shift = (target - raw) / n_atoms
        charges = [c + shift for c in charges]
        print(f"Net charge before correction: {raw:.8f}; library's (average, rounded) net charge: "
              f"{target}; shifting every atom by {shift:+.8f}.")

    # ---- Residues (support >1 residue in general, though these test
    # molecules are single-residue).
    residue_labels = []
    residue_pointers = []
    seen_res = None
    for aid in range(1, n_atoms + 1):
        res_key = (atoms[aid]["resname"], atoms[aid]["resnum"])
        if res_key != seen_res:
            residue_labels.append(atoms[aid]["resname"] or "MOL")
            residue_pointers.append(aid)
            seen_res = res_key

    # ---- Per-atom arrays.
    atom_names = [atoms[aid]["name"] for aid in range(1, n_atoms + 1)]
    atomic_numbers = [new_label[aid] for aid in range(1, n_atoms + 1)]
    masses = [assigned[aid]["mass"] for aid in range(1, n_atoms + 1)]
    atom_type_indices = [type_to_class[assigned_type[aid]] for aid in range(1, n_atoms + 1)]
    amber_types = [assigned_type[aid] for aid in range(1, n_atoms + 1)]
    radii = [assigned[aid]["radius"] for aid in range(1, n_atoms + 1)]
    screen = [assigned[aid]["screen"] for aid in range(1, n_atoms + 1)]
    tree = [assigned[aid]["tree"] for aid in range(1, n_atoms + 1)]
    join = [assigned[aid]["join"] for aid in range(1, n_atoms + 1)]
    irotat = [assigned[aid]["irotat"] for aid in range(1, n_atoms + 1)]

    natyp = len(set(amber_types))
    solty = [0.0] * natyp

    n_bond_h, n_bond_noh = len(bonds_h), len(bonds_noh)
    n_angle_h, n_angle_noh = len(angles_h), len(angles_noh)
    n_dih_h, n_dih_noh = len(dih_h), len(dih_noh)
    nnb = len(excluded_list)

    pointers = [
        n_atoms, ntypes, n_bond_h, n_bond_noh, n_angle_h, n_angle_noh,
        n_dih_h, n_dih_noh, 0, 0,
        nnb, len(residue_labels), n_bond_noh, n_angle_noh, n_dih_noh,
        len(bond_force_constants), len(angle_force_constants), len(dih_force_constants),
        natyp, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, max(number_excluded) if False else 0, 0,
        0, 0,
    ]
    # NMXRS (idx 28) is the max residue size; fill it in properly.
    res_sizes = []
    for idx in range(len(residue_pointers)):
        start = residue_pointers[idx]
        end = residue_pointers[idx + 1] if idx + 1 < len(residue_pointers) else n_atoms + 1
        res_sizes.append(end - start)
    pointers[28] = max(res_sizes) if res_sizes else n_atoms
    pointers = pointers[:31]
    while len(pointers) < 31:
        pointers.append(0)

    def flatten_bonds(entries):
        out = []
        for e in entries:
            out.extend(e)
        return out

    def flatten_angles(entries):
        out = []
        for e in entries:
            out.extend(e)
        return out

    def flatten_dih(entries):
        out = []
        for e in entries:
            out.extend(e)
        return out

    out = []
    out.append(f"%VERSION  VERSION_STAMP = V0001.000  DATE = 00/00/00  00:00:00  \n")
    write_section(out, "TITLE", [residue_labels[0] if residue_labels else "MOL"], 20, "a", 4, 0)
    write_section(out, "POINTERS", pointers, 10, "I", 8, 0)
    write_section(out, "ATOM_NAME", atom_names, 20, "a", 4, 0)
    write_section(out, "CHARGE", charges, 5, "E", 16, 8)
    write_section(out, "ATOMIC_NUMBER", atomic_numbers, 10, "I", 8, 0)
    write_section(out, "MASS", masses, 5, "E", 16, 8)
    write_section(out, "ATOM_TYPE_INDEX", atom_type_indices, 10, "I", 8, 0)
    write_section(out, "NUMBER_EXCLUDED_ATOMS", number_excluded, 10, "I", 8, 0)
    write_section(out, "NONBONDED_PARM_INDEX", ico, 10, "I", 8, 0)
    write_section(out, "RESIDUE_LABEL", residue_labels, 20, "a", 4, 0)
    write_section(out, "RESIDUE_POINTER", residue_pointers, 10, "I", 8, 0)
    write_section(out, "BOND_FORCE_CONSTANT", bond_force_constants, 5, "E", 16, 8)
    write_section(out, "BOND_EQUIL_VALUE", bond_equil_values, 5, "E", 16, 8)
    write_section(out, "ANGLE_FORCE_CONSTANT", angle_force_constants, 5, "E", 16, 8)
    write_section(out, "ANGLE_EQUIL_VALUE", angle_equil_values, 5, "E", 16, 8)
    write_section(out, "DIHEDRAL_FORCE_CONSTANT", dih_force_constants, 5, "E", 16, 8)
    write_section(out, "DIHEDRAL_PERIODICITY", dih_periodicities, 5, "E", 16, 8)
    write_section(out, "DIHEDRAL_PHASE", dih_phases, 5, "E", 16, 8)
    write_section(out, "SCEE_SCALE_FACTOR", dih_scee, 5, "E", 16, 8)
    write_section(out, "SCNB_SCALE_FACTOR", dih_scnb, 5, "E", 16, 8)
    write_section(out, "SOLTY", solty, 5, "E", 16, 8)
    write_section(out, "LENNARD_JONES_ACOEF", acoef, 5, "E", 16, 8)
    write_section(out, "LENNARD_JONES_BCOEF", bcoef, 5, "E", 16, 8)
    write_section(out, "BONDS_INC_HYDROGEN", flatten_bonds(bonds_h), 10, "I", 8, 0)
    write_section(out, "BONDS_WITHOUT_HYDROGEN", flatten_bonds(bonds_noh), 10, "I", 8, 0)
    write_section(out, "ANGLES_INC_HYDROGEN", flatten_angles(angles_h), 10, "I", 8, 0)
    write_section(out, "ANGLES_WITHOUT_HYDROGEN", flatten_angles(angles_noh), 10, "I", 8, 0)
    write_section(out, "DIHEDRALS_INC_HYDROGEN", flatten_dih(dih_h), 10, "I", 8, 0)
    write_section(out, "DIHEDRALS_WITHOUT_HYDROGEN", flatten_dih(dih_noh), 10, "I", 8, 0)
    write_section(out, "EXCLUDED_ATOMS_LIST", excluded_list, 10, "I", 8, 0)
    write_section(out, "HBOND_ACOEF", [], 5, "E", 16, 8)
    write_section(out, "HBOND_BCOEF", [], 5, "E", 16, 8)
    write_section(out, "HBCUT", [], 5, "E", 16, 8)
    write_section(out, "AMBER_ATOM_TYPE", amber_types, 20, "a", 4, 0)
    write_section(out, "TREE_CHAIN_CLASSIFICATION", tree, 20, "a", 4, 0)
    write_section(out, "JOIN_ARRAY", join, 10, "I", 8, 0)
    write_section(out, "IROTAT", irotat, 10, "I", 8, 0)
    write_section(out, "RADIUS_SET", [library.radius_set or "modified Bondi radii (mbondi)"], 1, "a", 80, 0)
    write_section(out, "RADII", radii, 5, "E", 16, 8)
    write_section(out, "SCREEN", screen, 5, "E", 16, 8)
    write_section(out, "IPOL", [0], 1, "I", 8, 0)

    with open(output_prmtop, "w") as f:
        f.writelines(out)

    print(f"Matched {len(assigned)}/{n_atoms} atom(s) to library fragments "
          f"({len(ambiguous)} ambiguous: {len(ambiguous) - len(majority_vote_atoms)} group-consistent, "
          f"{len(majority_vote_atoms)} majority-vote).")
    print(f"Wrote {output_prmtop}: {n_atoms} atoms, {ntypes} LJ types, "
          f"{n_bond_h + n_bond_noh} bonds, {n_angle_h + n_angle_noh} angles, "
          f"{n_dih_h + n_dih_noh} dihedral terms.", flush=True)


if __name__ == "__main__":
    main()
