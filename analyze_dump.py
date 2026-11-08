#!/usr/bin/env python3
"""
analyze_dump.py

Read a LAMMPS dump file (optionally zstd-compressed with a .zst extension)
and perform analysis on it. Currently reports the number of snapshots
(frames) contained in the dump file.

Usage:
    python3 analyze_dump.py dump.lammpstrj
    python3 analyze_dump.py dump.lammpstrj.zst
"""

import argparse
import math
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path


def open_dump(path):
    """Return a text-mode file-like object for a LAMMPS dump file.

    Transparently handles zstd-compressed files (.zst extension) by either
    using the `zstandard` python package if available, or falling back to
    piping through the `zstd` command line tool.
    """
    path = Path(path)

    if path.suffix != ".zst":
        return open(path, "r")

    # Try the python zstandard bindings first (avoids spawning a subprocess).
    try:
        import zstandard  # type: ignore

        fh = open(path, "rb")
        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        return io_text_wrapper(stream_reader, fh)
    except ImportError:
        pass

    # Fall back to the `zstd` CLI tool, streaming output through a pipe.
    zstd_exe = shutil.which("zstd")
    if zstd_exe is None:
        sys.exit(
            "Error: cannot read '{}'. Install the 'zstandard' python package "
            "(pip install zstandard) or the 'zstd' command line tool.".format(path)
        )

    proc = subprocess.Popen(
        [zstd_exe, "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return SubprocessTextReader(proc)


def io_text_wrapper(stream_reader, backing_fh):
    """Wrap a binary stream reader (zstandard) in a text wrapper, closing
    the underlying file handle when the wrapper is closed."""
    import io

    class _Wrapper(io.TextIOWrapper):
        def close(self):
            super().close()
            backing_fh.close()

    return _Wrapper(stream_reader, encoding="utf-8")


class SubprocessTextReader:
    """Minimal text-mode, line-iterable wrapper around a subprocess's stdout,
    used as a fallback when the `zstandard` python package is unavailable."""

    def __init__(self, proc):
        self._proc = proc
        self._text = io_wrap_binary(proc.stdout)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._text)

    def readline(self):
        return self._text.readline()

    def close(self):
        self._text.close()
        self._proc.stdout.close()
        self._proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def io_wrap_binary(binary_stream):
    import io

    return io.TextIOWrapper(binary_stream, encoding="utf-8")


def count_snapshots(dump_path):
    """Count the number of snapshots (frames) in a LAMMPS dump file.

    Each snapshot begins with an "ITEM: TIMESTEP" line, so counting those
    lines gives the number of frames without needing to fully parse the
    per-atom data.
    """
    count = 0
    with open_dump(dump_path) as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                count += 1
    return count


def find_last_timestep(dump_path):
    """Return the TIMESTEP value of the last snapshot in a dump file, by
    scanning for every "ITEM: TIMESTEP" line's value without parsing any
    per-atom data. Returns None if the dump file has no snapshots.
    """
    last_timestep = None
    with open_dump(dump_path) as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                last_timestep = int(next(fh).strip())
    return last_timestep


def _parse_box_bounds_line(line):
    """Parse a single BOX BOUNDS line (lo hi [tilt]) and return (lo, hi)."""
    tokens = line.split()
    return float(tokens[0]), float(tokens[1])


def read_snapshot(dump_path, target_timestep):
    """Scan a dump file for the snapshot with TIMESTEP == target_timestep.

    Returns a dict with keys "box" (list of (lo, hi) per dimension) and
    "atoms" (list of (atom_id, atom_type, atom_type_label, atom_mol, xs,
    ys, zs) tuples with the raw, scaled coordinates as found in the
    file), or None if the timestep was not found in the file.
    `atom_type` is the numeric type (int) if the dump file has a "type"
    column, else None; `atom_type_label` is the string type label if the
    dump file has a "typelabel" column, else None. At least one of the
    two columns must be present. `atom_mol` is the int molecule ID if
    the dump file has a "mol" column, else None.
    """
    with open_dump(dump_path) as fh:
        current_timestep = None
        box = None
        col_index = None
        atoms = []

        line = fh.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                current_timestep = int(fh.readline().strip())

            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(fh.readline().strip())

            elif line.startswith("ITEM: BOX BOUNDS"):
                box = [_parse_box_bounds_line(fh.readline()) for _ in range(3)]

            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]  # drop "ITEM:" and "ATOMS"
                col_index = {name: i for i, name in enumerate(headers)}
                for key in ("id", "xs", "ys", "zs"):
                    if key not in col_index:
                        sys.exit(
                            f"Error: dump file is missing required column '{key}' "
                            f"in 'ITEM: ATOMS' section (found columns: {headers})."
                        )
                if "type" not in col_index and "typelabel" not in col_index:
                    sys.exit(
                        "Error: dump file is missing a 'type' or 'typelabel' "
                        f"column in 'ITEM: ATOMS' section (found columns: {headers})."
                    )

                if current_timestep == target_timestep:
                    for _ in range(natoms):
                        tokens = fh.readline().split()
                        atom_id = int(tokens[col_index["id"]])
                        atom_type = (
                            int(tokens[col_index["type"]])
                            if "type" in col_index else None
                        )
                        atom_type_label = (
                            tokens[col_index["typelabel"]]
                            if "typelabel" in col_index else None
                        )
                        atom_mol = (
                            int(tokens[col_index["mol"]])
                            if "mol" in col_index else None
                        )
                        xs = float(tokens[col_index["xs"]])
                        ys = float(tokens[col_index["ys"]])
                        zs = float(tokens[col_index["zs"]])
                        atoms.append(
                            (atom_id, atom_type, atom_type_label, atom_mol, xs, ys, zs)
                        )
                    # Found and parsed the requested frame; no need to read further.
                    return {"box": box, "atoms": atoms}
                else:
                    # Not the frame we want; skip over its atom lines.
                    for _ in range(natoms):
                        fh.readline()

            line = fh.readline()

    return None


def iter_full_snapshots(dump_path, target_timestep=None):
    """Yield (timestep, box, atoms) for every snapshot in a dump file (or
    just the one matching `target_timestep`, if given).

    Unlike `read_snapshot` (which scans for a single target timestep and
    reads every column into a fixed-shape tuple), this reads *every*
    available per-atom column found in the dump file's "ITEM: ATOMS"
    header, generically, and is meant for streaming a whole trajectory
    (e.g. dump-to-xyz conversion) without re-scanning the file once per
    frame.

    Each yielded `atoms` is a list of per-atom dicts with at least "id",
    "xs", "ys", "zs" keys (raw, scaled coordinates), plus whichever of
    "mol", "type", "typelabel" columns are present in the file (as int,
    int, str respectively), and any other column found, left as a raw
    string. Missing optional columns are simply absent from each dict
    (use `.get(...)`).
    """
    with open_dump(dump_path) as fh:
        current_timestep = None
        box = None
        col_index = None

        line = fh.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                current_timestep = int(fh.readline().strip())

            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(fh.readline().strip())

            elif line.startswith("ITEM: BOX BOUNDS"):
                box = [_parse_box_bounds_line(fh.readline()) for _ in range(3)]

            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]  # drop "ITEM:" and "ATOMS"
                col_index = {name: i for i, name in enumerate(headers)}
                for key in ("id", "xs", "ys", "zs"):
                    if key not in col_index:
                        sys.exit(
                            f"Error: dump file is missing required column '{key}' "
                            f"in 'ITEM: ATOMS' section (found columns: {headers})."
                        )

                is_wanted_frame = (
                    target_timestep is None or current_timestep == target_timestep
                )
                if is_wanted_frame:
                    atoms = []
                    for _ in range(natoms):
                        tokens = fh.readline().split()
                        atom = {
                            "id": int(tokens[col_index["id"]]),
                            "xs": float(tokens[col_index["xs"]]),
                            "ys": float(tokens[col_index["ys"]]),
                            "zs": float(tokens[col_index["zs"]]),
                        }
                        if "mol" in col_index:
                            atom["mol"] = int(tokens[col_index["mol"]])
                        if "type" in col_index:
                            atom["type"] = int(tokens[col_index["type"]])
                        if "typelabel" in col_index:
                            atom["typelabel"] = tokens[col_index["typelabel"]]
                        for name, idx in col_index.items():
                            if name not in atom:
                                atom[name] = tokens[idx]
                        atoms.append(atom)
                    yield current_timestep, box, atoms
                    if target_timestep is not None:
                        return
                else:
                    for _ in range(natoms):
                        fh.readline()

            line = fh.readline()


def scaled_to_real(xs, ys, zs, box):
    """Convert scaled (fractional) coordinates in [0, 1] to real coordinates
    using the per-dimension (lo, hi) box bounds."""
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
    x = xlo + xs * (xhi - xlo)
    y = ylo + ys * (yhi - ylo)
    z = zlo + zs * (zhi - zlo)
    return x, y, z


def minimum_image_distance(pos1, pos2, box_lengths):
    """Distance between two real-coordinate points under the minimum image
    convention for an orthogonal, periodic box."""
    total = 0.0
    for coord1, coord2, length in zip(pos1, pos2, box_lengths):
        delta = coord1 - coord2
        if length > 0:
            delta -= length * round(delta / length)
        total += delta * delta
    return math.sqrt(total)


def iter_snapshot_positions(dump_path, atom_ids, target_timestep=None):
    """Yield (timestep, box, positions) for every snapshot in the dump file
    (or just the one matching `target_timestep`, if given), where `positions`
    is a dict mapping requested atom_id -> real-coordinate (x, y, z) tuple.

    Only the atoms listed in `atom_ids` are extracted/converted, so this
    stays cheap even for large frames. Atom IDs missing from a given
    snapshot are simply absent from that frame's `positions` dict.
    """
    wanted_ids = set(atom_ids)
    with open_dump(dump_path) as fh:
        current_timestep = None
        box = None
        col_index = None

        line = fh.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                current_timestep = int(fh.readline().strip())

            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                natoms = int(fh.readline().strip())

            elif line.startswith("ITEM: BOX BOUNDS"):
                box = [_parse_box_bounds_line(fh.readline()) for _ in range(3)]

            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]
                col_index = {name: i for i, name in enumerate(headers)}
                for key in ("id", "xs", "ys", "zs"):
                    if key not in col_index:
                        sys.exit(
                            f"Error: dump file is missing required column '{key}' "
                            f"in 'ITEM: ATOMS' section (found columns: {headers})."
                        )

                is_wanted_frame = (
                    target_timestep is None or current_timestep == target_timestep
                )
                if is_wanted_frame:
                    positions = {}
                    for _ in range(natoms):
                        tokens = fh.readline().split()
                        atom_id = int(tokens[col_index["id"]])
                        if atom_id in wanted_ids:
                            xs = float(tokens[col_index["xs"]])
                            ys = float(tokens[col_index["ys"]])
                            zs = float(tokens[col_index["zs"]])
                            positions[atom_id] = scaled_to_real(xs, ys, zs, box)

                    yield current_timestep, box, positions

                    if target_timestep is not None:
                        return
                else:
                    for _ in range(natoms):
                        fh.readline()

            line = fh.readline()


def iter_atom_pair_distances(dump_path, atom_id1, atom_id2, target_timestep=None):
    """Yield (timestep, distance) for a specific pair of atom IDs across every
    snapshot in the dump file (or just the one matching `target_timestep`, if
    given). Frames where either atom ID is missing are skipped with a
    warning printed to stderr.
    """
    for timestep, box, positions in iter_snapshot_positions(
        dump_path, (atom_id1, atom_id2), target_timestep
    ):
        if atom_id1 not in positions or atom_id2 not in positions:
            missing = atom_id1 if atom_id1 not in positions else atom_id2
            print(
                f"Warning: atom {missing} not found in snapshot {timestep}; skipping.",
                file=sys.stderr,
            )
            continue

        box_lengths = [hi - lo for lo, hi in box]
        dist = minimum_image_distance(
            positions[atom_id1], positions[atom_id2], box_lengths
        )
        yield timestep, dist


def minimum_image_vector(pos_from, pos_to, box_lengths):
    """Displacement vector from pos_from to pos_to under the minimum image
    convention for an orthogonal, periodic box."""
    delta = []
    for c_from, c_to, length in zip(pos_from, pos_to, box_lengths):
        d = c_to - c_from
        if length > 0:
            d -= length * round(d / length)
        delta.append(d)
    return delta


def iter_atom_triplet_angles(
    dump_path, atom_id1, atom_id2, atom_id3, target_timestep=None
):
    """Yield (timestep, angle_degrees) for the angle at atom_id2 (the vertex)
    formed by atom_id1-atom_id2-atom_id3, across every snapshot in the dump
    file (or just the one matching `target_timestep`, if given). The two
    bond vectors (vertex -> atom_id1 and vertex -> atom_id3) are each wrapped
    independently using the minimum image convention before the angle is
    computed. Frames where any of the three atom IDs are missing are skipped
    with a warning printed to stderr.
    """
    atom_ids = (atom_id1, atom_id2, atom_id3)
    for timestep, box, positions in iter_snapshot_positions(
        dump_path, atom_ids, target_timestep
    ):
        missing = [a for a in atom_ids if a not in positions]
        if missing:
            print(
                f"Warning: atom(s) {missing} not found in snapshot {timestep}; "
                "skipping.",
                file=sys.stderr,
            )
            continue

        box_lengths = [hi - lo for lo, hi in box]
        vertex = positions[atom_id2]
        v1 = minimum_image_vector(vertex, positions[atom_id1], box_lengths)
        v2 = minimum_image_vector(vertex, positions[atom_id3], box_lengths)

        norm1 = math.sqrt(sum(c * c for c in v1))
        norm2 = math.sqrt(sum(c * c for c in v2))
        if norm1 == 0 or norm2 == 0:
            print(
                f"Warning: degenerate geometry (zero-length bond) in snapshot "
                f"{timestep}; skipping.",
                file=sys.stderr,
            )
            continue

        dot = sum(c1 * c2 for c1, c2 in zip(v1, v2))
        cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        angle = math.degrees(math.acos(cos_theta))
        yield timestep, angle


def count_molecule_sizes(dump_path, target_timestep):
    """Count how many atoms belong to each molecule in a snapshot, then
    tally how many molecules have each atom count.

    Returns a dict mapping molecule size (int, number of atoms) to the
    number of molecules of that size, e.g. {3: 10, 5: 2} means 10
    molecules with 3 atoms each and 2 molecules with 5 atoms each.
    """
    snapshot = read_snapshot(dump_path, target_timestep)
    if snapshot is None:
        sys.exit(f"Error: snapshot with timestep {target_timestep} not found.")

    atoms_per_mol = {}
    for atom_id, atom_type, atom_type_label, atom_mol, xs, ys, zs in snapshot["atoms"]:
        if atom_mol is None:
            sys.exit(
                "Error: dump file has no 'mol' column, cannot determine molecule sizes."
            )
        atoms_per_mol[atom_mol] = atoms_per_mol.get(atom_mol, 0) + 1

    size_counts = {}
    for n_atoms in atoms_per_mol.values():
        size_counts[n_atoms] = size_counts.get(n_atoms, 0) + 1

    return size_counts


def _parse_type_arg(value):
    """Parse a --types command-line value: an int if it looks numeric,
    otherwise the raw string (a type label, matched against the dump
    file's "typelabel" column).
    """
    try:
        return int(value)
    except ValueError:
        return value


def _parse_snapshot_arg(value):
    """argparse `type=` helper for -s/--snapshot: accepts either an int
    TIMESTEP or the special value "last" (case-insensitive), returned
    as-is (the literal string "last") for the caller to resolve against
    a specific dump file with find_last_timestep, once it knows which
    dump file that is.
    """
    if value.lower() == "last":
        return "last"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid snapshot '{value}' (expected an integer TIMESTEP or "
            "'last')"
        )


def find_close_pairs(dump_path, target_timestep, type1, type2, cutoff):
    """Find pairs of atoms of the given types in the given snapshot that are
    within `cutoff` (real-units) of each other, accounting for periodic
    boundary conditions via the minimum image convention.

    `type1`/`type2` are each either an int (matched against the dump
    file's numeric "type" column) or a str (matched against its
    "typelabel" column, e.g. as found in a dump written with LAMMPS type
    labels).

    Returns a list of (atom_id1, mol1, atom_id2, mol2, distance) tuples,
    sorted by ascending distance. `mol1`/`mol2` are the molecule IDs of
    each atom (int) if the dump file has a "mol" column, else None.
    """
    snapshot = read_snapshot(dump_path, target_timestep)
    if snapshot is None:
        sys.exit(f"Error: timestep {target_timestep} not found in '{dump_path}'.")

    box = snapshot["box"]
    box_lengths = [hi - lo for lo, hi in box]

    def matches(atom_type, atom_type_label, type_arg):
        if isinstance(type_arg, int):
            if atom_type is None:
                sys.exit(
                    f"Error: type {type_arg} is numeric, but '{dump_path}' has "
                    "no 'type' column (only 'typelabel'); use a type label "
                    "instead."
                )
            return atom_type == type_arg
        else:
            if atom_type_label is None:
                sys.exit(
                    f"Error: type '{type_arg}' is not numeric, but "
                    f"'{dump_path}' has no 'typelabel' column (only 'type'); "
                    "use a numeric type instead."
                )
            return atom_type_label == type_arg

    group1, group2 = [], []
    for atom_id, atom_type, atom_type_label, atom_mol, xs, ys, zs in snapshot["atoms"]:
        is_type1 = matches(atom_type, atom_type_label, type1)
        is_type2 = matches(atom_type, atom_type_label, type2)
        if not (is_type1 or is_type2):
            continue
        pos = scaled_to_real(xs, ys, zs, box)
        if is_type1:
            group1.append((atom_id, atom_mol, pos))
        if is_type2:
            group2.append((atom_id, atom_mol, pos))

    pairs = []
    if type1 == type2:
        candidate_pairs = combinations(group1, 2)
    else:
        candidate_pairs = ((a, b) for a in group1 for b in group2)

    seen = set()
    for (id1, mol1, pos1), (id2, mol2, pos2) in candidate_pairs:
        if id1 == id2:
            continue
        key = (min(id1, id2), max(id1, id2))
        if key in seen:
            continue
        seen.add(key)
        dist = minimum_image_distance(pos1, pos2, box_lengths)
        if dist < cutoff:
            pairs.append((id1, mol1, id2, mol2, dist))

    pairs.sort(key=lambda p: p[-1])
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a LAMMPS dump file (plain text or .zst compressed)."
    )
    parser.add_argument("dump_file", help="Path to the LAMMPS dump file to analyze.")
    parser.add_argument(
        "-s",
        "--snapshot",
        type=_parse_snapshot_arg,
        default=None,
        metavar="TIMESTEP",
        help="TIMESTEP value of the snapshot to analyze (as recorded in "
        "the dump file), or the special value 'last' to select the final "
        "snapshot in the dump file.",
    )
    parser.add_argument(
        "-t",
        "--types",
        type=_parse_type_arg,
        nargs=2,
        metavar=("TYPE1", "TYPE2"),
        default=None,
        help="Pair of atom types to search for close contacts: numeric "
        "types (matched against the dump file's 'type' column), or, if "
        "not numeric, type labels (matched against its 'typelabel' "
        "column, e.g. `ca hn`).",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=float,
        default=None,
        help="Distance cutoff (real units, same as the box bounds) for close atom pairs.",
    )
    parser.add_argument(
        "-a",
        "--atoms",
        type=int,
        nargs=2,
        metavar=("ID1", "ID2"),
        default=None,
        help=(
            "Pair of atom IDs (1st column of ITEM: ATOMS) to report the "
            "distance between for each snapshot (or just --snapshot, if given)."
        ),
    )
    parser.add_argument(
        "-g",
        "--angle",
        type=int,
        nargs=3,
        metavar=("ID1", "ID2", "ID3"),
        default=None,
        help=(
            "Triplet of atom IDs to report the angle (in degrees) for, "
            "at each snapshot (or just --snapshot, if given). The angle is "
            "measured at ID2 (the vertex), i.e. the angle ID1-ID2-ID3."
        ),
    )
    parser.add_argument(
        "-m",
        "--molecule-sizes",
        action="store_true",
        help=(
            "Report how many molecules of each size (atom count) are "
            "present in the given --snapshot."
        ),
    )
    args = parser.parse_args()

    dump_path = Path(args.dump_file)
    if not dump_path.is_file():
        sys.exit(f"Error: file not found: {dump_path}")

    if args.snapshot == "last":
        args.snapshot = find_last_timestep(dump_path)
        if args.snapshot is None:
            sys.exit(f"Error: '{dump_path}' contains no snapshots.")

    pair_search_args = (args.types, args.distance)
    do_type_pair_search = any(v is not None for v in pair_search_args)
    do_atom_pair_report = args.atoms is not None
    do_angle_report = args.angle is not None
    do_molecule_size_report = args.molecule_sizes

    exclusive_modes = [
        do_type_pair_search,
        do_atom_pair_report,
        do_angle_report,
        do_molecule_size_report,
    ]
    if sum(exclusive_modes) > 1:
        sys.exit(
            "Error: --types/--distance, --atoms, --angle, and "
            "--molecule-sizes are mutually exclusive; use only one at a time."
        )

    if do_type_pair_search:
        if args.snapshot is None or any(v is None for v in pair_search_args):
            sys.exit(
                "Error: --snapshot, --types, and --distance must all be given "
                "together to search for close atom pairs."
            )

        type1, type2 = args.types
        pairs = find_close_pairs(
            dump_path, args.snapshot, type1, type2, args.distance
        )
        print(
            f"Snapshot {args.snapshot}: found {len(pairs)} pair(s) of type "
            f"{type1}-{type2} atoms within {args.distance} of each other:"
        )
        for id1, mol1, id2, mol2, dist in pairs:
            mol1_str = "?" if mol1 is None else str(mol1)
            mol2_str = "?" if mol2 is None else str(mol2)
            print(
                f"  atom {id1} (mol {mol1_str}) -- atom {id2} (mol {mol2_str}): "
                f"distance = {dist:.6f}"
            )

    elif do_atom_pair_report:
        atom_id1, atom_id2 = args.atoms
        found_any = False
        for timestep, dist in iter_atom_pair_distances(
            dump_path, atom_id1, atom_id2, target_timestep=args.snapshot
        ):
            found_any = True
            print(f"Timestep {timestep}: distance(atom {atom_id1}, atom {atom_id2}) = {dist:.6f}")
        if not found_any:
            sys.exit(
                f"Error: no snapshot contained both atom {atom_id1} and atom {atom_id2}."
            )

    elif do_angle_report:
        atom_id1, atom_id2, atom_id3 = args.angle
        found_any = False
        for timestep, angle in iter_atom_triplet_angles(
            dump_path, atom_id1, atom_id2, atom_id3, target_timestep=args.snapshot
        ):
            found_any = True
            print(
                f"Timestep {timestep}: angle(atom {atom_id1}, atom {atom_id2}, "
                f"atom {atom_id3}) = {angle:.4f} degrees"
            )
        if not found_any:
            sys.exit(
                f"Error: no snapshot contained atoms {atom_id1}, {atom_id2}, "
                f"and {atom_id3}."
            )

    elif do_molecule_size_report:
        if args.snapshot is None:
            sys.exit("Error: --snapshot must be given with --molecule-sizes.")

        size_counts = count_molecule_sizes(dump_path, args.snapshot)
        n_molecules = sum(size_counts.values())
        print(
            f"Snapshot {args.snapshot}: {n_molecules} molecule(s), "
            f"molecule sizes (atom count -> number of molecules):"
        )
        for size in sorted(size_counts):
            print(f"  {size} atoms: {size_counts[size]} molecule(s)")

    else:
        n_snapshots = count_snapshots(dump_path)
        print(f"Number of snapshots in '{dump_path}': {n_snapshots}")


if __name__ == "__main__":
    main()
