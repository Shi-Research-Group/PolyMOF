#!/usr/bin/env python3
"""
dump2xyz.py

Convert a LAMMPS dump file (optionally zstd-compressed with a .zst
extension) into a standard (possibly multi-frame) XYZ file, or a single
PDB file, reusing analyze_dump.py's dump-reading machinery.

The second command-line argument is the output file to write; its type
is determined by its extension: a ".pdb" extension writes a PDB file,
and anything else (e.g. ".xyz") writes an XYZ file. Writing a PDB file
requires a single snapshot to be selected with -s/--snapshot, since the
PDB format written here does not support multiple frames. -s/--snapshot
also accepts the special value "last" to select the final snapshot in
the dump file, instead of a specific TIMESTEP.

Each XYZ frame is written as:
    <natoms>
    Timestep <timestep>
    <element> <x> <y> <z>
    ...
in real (not scaled) coordinates. The "element" column is the atom's
type label (if the dump file has a "typelabel" column, e.g. one written
with LAMMPS type labels) or, failing that, its numeric type.

The PDB file is written as a single MODEL with one CRYST1 record (from
the snapshot's box bounds, assumed orthogonal) and one ATOM record per
atom, again in real coordinates. As with the XYZ output, the atom name
column is filled in with the atom's type label or numeric type -- not
necessarily a valid chemical element symbol -- since that is all the
dump file provides. The element column is filled in with a guessed
one-character element symbol: the uppercased first character of the
atom's type label/name.

Optionally, --atoms restricts the output to only the atom(s) with the
given atom ID(s) and any other atoms that share a molecule with one of
them, in each frame -- i.e. it is resolved to a set of molecule IDs
independently for every frame (since an atom's molecule assignment can
change over the course of the dump, e.g. across a reaction), rather
than being fixed once at the start. Requires the dump file to have a
"mol" column. Every other atom is left out of the output file entirely.

Optionally, --pbc applies periodic boundary conditions to wrap each
atom's scaled coordinates back into the box before converting to real
coordinates, so every written coordinate falls in [0, hi - lo) along
each dimension -- i.e. the box is shifted to start at the origin,
rather than at (xlo, ylo, zlo) -- instead of the default of simply
scaling each atom's (possibly out-of-box, unwrapped) coordinates
relative to the dump file's own (lo, hi) box bounds.

Optionally, --data-file points to a LAMMPS data file (e.g. the one the
dump file's run started from) to take a bond list from. When given and
a PDB file is being written, its "Bonds" section is used to write
CONECT records for every bond whose two atom IDs are both present in
the PDB file's output (matched by atom ID, so the data file's atom
numbering must agree with the dump file's); bonds involving an atom
filtered out (e.g. via --atoms) are simply skipped. Ignored when
writing an XYZ file, since the XYZ format has no way to record bonds.

Usage:
    python3 dump2xyz.py dump.lammpstrj out.xyz
    python3 dump2xyz.py dump.lammpstrj.zst out.xyz --snapshot 1000
    python3 dump2xyz.py dump2.lammpstrj out.xyz --atoms 5,12,30
    python3 dump2xyz.py dump.lammpstrj.zst out.pdb --snapshot 1000
    python3 dump2xyz.py dump.lammpstrj.zst out.xyz --pbc
    python3 dump2xyz.py dump.lammpstrj.zst out.pdb -s 1100000 \
        --data-file anneal-data.lammps
    python3 dump2xyz.py dump.lammpstrj.zst out.pdb -s last
"""

import argparse
import sys
from pathlib import Path
import math

import plot_molecule_graph as pmg

import analyze_dump as ad

from operator import itemgetter

def get_atom_columns(dump_path):
    """Return the list of per-atom column names from the first 'ITEM:
    ATOMS' header found in the dump file, without reading the rest of it.
    """
    with ad.open_dump(dump_path) as fh:
        for line in fh:
            if line.startswith("ITEM: ATOMS"):
                return line.split()[2:]
    return []


def parse_data_file_bonds(data_path):
    """Parse a LAMMPS data file's "Bonds" section and return a list of
    (atom_id1, atom_id2) int tuples, one per bond, reusing
    plot_molecule_graph.py's generic LAMMPS section parser (every other
    section of the data file, e.g. "Atoms", is ignored).
    """
    sections = pmg.parse_sections(data_path)
    if "Bonds" not in sections:
        sys.exit(f"Error: no 'Bonds' section found in '{data_path}'.")
    return pmg.parse_bonds(sections)


def atom_label(atom):
    """Return the atom's "element" field for the XYZ/PDB file: its type
    label if the dump file has one, else its numeric type.
    """
    if "typelabel" in atom:
        return atom["typelabel"]
    if "type" in atom:
        return str(atom["type"])
    return "X"


def _require_mol_column(dump_path, atom_ids):
    """Exit with an error if `atom_ids` is given but `dump_path` has no
    'mol' column (--atoms needs one to resolve molecule membership).
    """
    if atom_ids is not None:
        columns = get_atom_columns(dump_path)
        if "mol" not in columns:
            sys.exit(
                f"Error: --atoms requires a 'mol' column in "
                f"'{dump_path}' (found columns: {columns})."
            )


def _filter_atoms(atoms, atom_ids):
    """Return only the atoms in `atoms` that have an ID in `atom_ids`.
    """
    if atom_ids is None:
        return atoms
    return [a for a in atoms if str(a['id']) in atom_ids]

def _filter_atoms_by_mol(atoms, atom_ids):
    """Return only the atoms in `atoms` that either have an ID in
    `atom_ids` or share a "mol" value with one that does (or `atoms`
    unchanged if `atom_ids` is None).
    """
    if atom_ids is None:
        return atoms
    target_mols = {a["mol"] for a in atoms if a["id"] in atom_ids and "mol" in a}
    return [a for a in atoms if a.get("mol") in target_mols]


def _distance_squared(atom1, atom2, box, pbc):
    """Calculate squared distance between two atoms in real coordinates.
    If pbc is True, account for periodic boundary conditions.
    """
    x1, y1, z1 = _atom_real_coords(atom1, box, pbc)
    x2, y2, z2 = _atom_real_coords(atom2, box, pbc)
    
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    
    if pbc and box is not None:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
        xlen = xhi - xlo
        ylen = yhi - ylo
        zlen = zhi - zlo
        
        # Apply minimum image convention
        if abs(dx) > xlen / 2:
            dx = xlen - abs(dx) if dx > 0 else -xlen + abs(dx)
        if abs(dy) > ylen / 2:
            dy = ylen - abs(dy) if dy > 0 else -ylen + abs(dy)
        if abs(dz) > zlen / 2:
            dz = zlen - abs(dz) if dz > 0 else -zlen + abs(dz)
    
    return dx*dx + dy*dy + dz*dz


def _filter_atoms_by_distance(atoms, atom_ids, box, pbc, distance_cutoff):
    """Return atoms that either have an ID in atom_ids or are within
    distance_cutoff of any atom with an ID in atom_ids. If atom_ids is
    None or distance_cutoff is None, return atoms unchanged.
    """
    if atom_ids is None or distance_cutoff is None:
        return atoms
    
    atom_ids_set = set(atom_ids)
    target_atoms = [a for a in atoms if a["id"] in atom_ids_set]
    
    if not target_atoms:
        return atoms
    
    cutoff_squared = distance_cutoff * distance_cutoff
    included_ids = set(atom_ids)
    
    for atom in atoms:
        if atom["id"] not in included_ids:
            for target_atom in target_atoms:
                dist_sq = _distance_squared(atom, target_atom, box, pbc)
                if dist_sq <= cutoff_squared:
                    included_ids.add(atom["id"])
                    break
    
    return [a for a in atoms if a["id"] in included_ids]



def type2element(atom):
    """Return the atom's guessed one-character chemical element symbol for
    the PDB file's element column: the uppercased first character of its
    type label/name (see `atom_label`).
    """
    label = atom_label(atom)
    if label in ["zn","fe"]:
        return label.upper()
    return label[0].upper() if label else "X"


def _atom_real_coords(atom, box, pbc, center=None):
    """Return an atom's (x, y, z) real coordinates from its scaled "xs",
    "ys", "zs" fields and the frame's `box` bounds. If `pbc` is true, the
    scaled coordinates are first wrapped (mod 1) back into [0, 1) along
    each dimension before scaling, so the written coordinates land in
    [0, hi - lo) - i.e. the box origin is shifted to (0, 0, 0) - rather
    than being scaled relative to the dump file's own (lo, hi) bounds
    (which can leave unwrapped atoms outside the box, e.g. when the box
    size changes over the course of a dump).
    """
    xs, ys, zs = (atom["xs"], atom["ys"], atom["zs"])
    if center is not None:
        xs += center[0]
        ys += center[1]
        zs += center[2]
    if pbc:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
        x = (xs % 1.0) * (xhi - xlo)
        y = (ys % 1.0) * (yhi - ylo)
        z = (zs % 1.0) * (zhi - zlo)
        return x, y, z
    return ad.scaled_to_real(xs, ys, zs, box)


def write_xyz(dump_path, xyz_path, atom_ids=None, exact_atoms=True, target_timestep=None, pbc=False, center=None, distance_cutoff=None):
    """Convert `dump_path` to a (possibly multi-frame) XYZ file at
    `xyz_path`. If `atom_ids` is given (an iterable of int atom IDs),
    then for every frame, only the atom(s) with those IDs and any other
    atoms sharing a molecule with one of them (per that frame's "mol"
    column values -- re-resolved independently each frame, since an
    atom's molecule assignment can change over the dump) are written.
    If `distance_cutoff` is given, also include any atoms within that
    distance of the specified atoms (checked for each frame). If
    `target_timestep` is given, only that snapshot is converted. If
    `pbc` is true, each atom's coordinates are wrapped back into the box
    (see `_atom_real_coords`) before being written.
    """
    _require_mol_column(dump_path, atom_ids)
    atom_ids = set(atom_ids) if atom_ids is not None else None

    n_frames = 0
    with open(xyz_path, "w") as out:
        for timestep, box, atoms in ad.iter_full_snapshots(
            dump_path, target_timestep=target_timestep
        ):
            xyzshift = None
            if center is not None:
                center_atom = _filter_atoms(atoms, [center])[0]
                xyzshift = (0.5 - center_atom["xs"], 0.5 - center_atom["ys"], 0.5 - center_atom["zs"])
            if distance_cutoff is not None:
                atoms = _filter_atoms_by_distance(atoms, atom_ids, box, pbc, distance_cutoff)
            elif exact_atoms:
                atoms = _filter_atoms(atoms, atom_ids)
            else:
                atoms = _filter_atoms_by_mol(atoms, atom_ids)
            atoms.sort(key=itemgetter('id'))

            out.write(f"{len(atoms)}\n")
            out.write(f"Timestep {timestep}\n")
            for atom in atoms:
                x, y, z = _atom_real_coords(atom, box, pbc, xyzshift)
                out.write(f"{type2element(atom)} {x:.6f} {y:.6f} {z:.6f}\n")
            n_frames += 1

    if n_frames == 0:
        sys.exit(
            f"Error: timestep {target_timestep} not found in '{dump_path}'."
            if target_timestep is not None
            else f"Error: '{dump_path}' contains no snapshots."
        )

    return n_frames


def write_pdb(
    dump_path, pdb_path, atom_ids=None, exact_atoms = True, target_timestep=None, pbc=False, center=None,
    bonds=None, distance_cutoff=None,
):
    """Convert a single snapshot of `dump_path` to a PDB file at
    `pdb_path`. `target_timestep` must be given (the PDB format written
    here holds only one frame/MODEL), and selects which snapshot to
    convert. If `atom_ids` is given (an iterable of int atom IDs), only
    the atom(s) with those IDs and any other atoms sharing a molecule
    with one of them (per the snapshot's "mol" column values) are
    written. If `distance_cutoff` is given, also include any atoms
    within that distance of the specified atoms. If `pbc` is true, each
    atom's coordinates are wrapped back into the box (see
    `_atom_real_coords`) before being written.

    The snapshot's (assumed orthogonal) box bounds are written as a
    CRYST1 record; each atom is written as an ATOM record with its name
    column filled in the same way as write_xyz's per-atom label (the
    atom's type label or numeric type -- not necessarily a valid
    chemical element symbol), and its element column filled in with a
    guessed one-character element symbol (see `type2element`).

    If `bonds` is given (an iterable of (atom_id1, atom_id2) int
    tuples, e.g. from parse_data_file_bonds), a CONECT record is
    written for every bonded atom pair that both end up in the output
    (matched by atom ID); bonds involving a filtered-out atom are
    skipped.
    """
    assert target_timestep is not None, "write_pdb requires target_timestep"

    _require_mol_column(dump_path, atom_ids)
    atom_ids = set(atom_ids) if atom_ids is not None else None

    frame = next(
        ad.iter_full_snapshots(dump_path, target_timestep=target_timestep), None
    )
    if frame is None:
        sys.exit(f"Error: timestep {target_timestep} not found in '{dump_path}'.")
    timestep, box, atoms = frame
    if distance_cutoff is not None:
        atoms = _filter_atoms_by_distance(atoms, atom_ids, box, pbc, distance_cutoff)
    elif exact_atoms:
        atoms = _filter_atoms(atoms, atom_ids)
    else:
        atoms = _filter_atoms_by_mol(atoms, atom_ids)

    serial_by_id = {}
    with open(pdb_path, "w") as out:
        out.write(f"REMARK Timestep {timestep}\n")
        if box is not None:
            (xlo, xhi), (ylo, yhi), (zlo, zhi) = box
            out.write(
                "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} {:<11s}{:4d}\n".format(
                    xhi - xlo, yhi - ylo, zhi - zlo, 90.0, 90.0, 90.0, "P 1", 1
                )
            )
        xyzshift = None
        if center is not None:
            center_atom = _filter_atoms(atoms, [center])[0]
            xyzshift = (0.5 - center_atom["xs"], 0.5 - center_atom["ys"], 0.5 - center_atom["zs"])
        for i, atom in enumerate(atoms, start=1):
            x, y, z = _atom_real_coords(atom, box, pbc, xyzshift)
            name = atom_label(atom)
            resseq = atom.get("mol", 1)
            serial = i % 100000
            serial_by_id[atom["id"]] = serial
            out.write(
                "ATOM  {:5d} {:<4s} MOL A{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    serial, name[:4], resseq % 10000, x, y, z, 1.0, 0.0,
                    type2element(atom),
                )
            )

        if bonds is not None:
            _write_conect_records(out, bonds, serial_by_id)

        out.write("END\n")

    return 1


def _write_conect_records(out, bonds, serial_by_id):
    """Write CONECT records to `out` for every bond in `bonds` (an
    iterable of (atom_id1, atom_id2) int tuples) whose both atom IDs are
    keys of `serial_by_id` (atom_id -> PDB serial number), in atom-ID
    order. Each bonded neighbor is listed on both atoms' CONECT records
    (as is conventional in PDB files), and an atom with more than 4
    bonded neighbors gets multiple CONECT records (the PDB format allows
    at most 4 bonded-atom serials per record).
    """
    neighbors_by_id = {}
    for a, b in bonds:
        if a not in serial_by_id or b not in serial_by_id:
            continue
        neighbors_by_id.setdefault(a, []).append(b)
        neighbors_by_id.setdefault(b, []).append(a)

    for atom_id in sorted(neighbors_by_id, key=lambda aid: serial_by_id[aid]):
        neighbor_serials = [
            serial_by_id[nid] for nid in neighbors_by_id[atom_id]
        ]
        serial = serial_by_id[atom_id]
        for chunk_start in range(0, len(neighbor_serials), 4):
            chunk = neighbor_serials[chunk_start:chunk_start + 4]
            out.write(
                "CONECT{:5d}".format(serial)
                + "".join(f"{s:5d}" for s in chunk)
                + "\n"
            )

def read_atoms_file(atoms_file):
    atoms = []
    with open(atoms_file, "r") as f:
        for l in f:
            bits = l.split()
            atoms += bits
    return atoms

def _parse_atom_id_list(value):
    """argparse `type=` helper for --atoms: parses a comma-separated (and/or
    whitespace-separated) list of int atom IDs from a single token, e.g.
    "4,1278,328". This lets --atoms accept a single argument regardless
    of where it falls relative to the positional dump_file/xyz_file
    arguments, avoiding the ambiguity of an `nargs="+"` option greedily
    consuming unrelated positional arguments that happen to follow it.
    """
    try:
        return [int(tok) for tok in value.replace(",", " ").split()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid atom ID list: '{value}' (expected comma- or "
            "space-separated integers)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a LAMMPS dump file (plain text or .zst "
        "compressed) into a standard XYZ or PDB file, according to the "
        "output file's extension."
    )
    parser.add_argument("dump_file", help="Path to the LAMMPS dump file to convert.")
    parser.add_argument(
        "out_file",
        help="Path to the output file to write. Its extension determines "
        "the file type written: '.pdb' writes a PDB file (which requires "
        "-s/--snapshot, since only one frame can be written), and any "
        "other extension (e.g. '.xyz') writes an XYZ file.",
    )
    parser.add_argument(
        "-s", "--snapshot", type=ad._parse_snapshot_arg, default=None,
        metavar="TIMESTEP",
        help="Only convert the snapshot with this TIMESTEP (as recorded in "
        "the dump file), or, if given the special value 'last', the final "
        "snapshot in the dump file. Default: convert every snapshot, as a "
        "multi-frame XYZ file. Required when writing a PDB output file.",
    )
    parser.add_argument(
        "--atoms-file", 
        help="A file with a list of atom IDs.  Only atoms listed will "
        "be included in the output."
    )
    parser.add_argument(
        "-a", "--atoms", type=_parse_atom_id_list, action="extend",
        default=None, metavar="ATOM_IDS",
        help="Only include the atom(s) with these ID(s) and any other "
        "atoms sharing a molecule with one of them, matched against the "
        "dump file's 'mol' column (requires the dump file to have one). "
        "Molecule membership is resolved independently for each frame, "
        "since it can change over the course of the dump (e.g. across a "
        "reaction). Give IDs as a single comma-separated argument (e.g. "
        "-a 4,1278,328,708,860); this option can be repeated. Default: "
        "include every atom.",
    )
    parser.add_argument(
        "--pbc", action="store_true",
        help="Apply periodic boundary conditions: wrap each atom's scaled "
        "coordinates (mod 1) back into the box before converting to real "
        "coordinates, so every written coordinate falls in [0, hi - lo) "
        "along each dimension (i.e. the box is shifted to start at the "
        "origin, rather than at (xlo, ylo, zlo)). Default: scale each "
        "atom's coordinates relative to the dump file's own (lo, hi) box "
        "bounds as-is, without wrapping.",
    )
    parser.add_argument(
        "-d", "--data-file", default=None, metavar="DATA_FILE",
        help="Path to a LAMMPS data file (e.g. the one the dump file's run "
        "started from) to take a bond list from. Only used when writing a "
        "PDB output file: its 'Bonds' section is used to write CONECT "
        "records for every bond whose two atom IDs are both present in "
        "the output (matched by atom ID against the dump file's own atom "
        "IDs). Ignored when writing an XYZ file. Default: no bond list, "
        "so no CONECT records are written.",
    )
    parser.add_argument(
        "--distance", type=float, default=None, metavar="DISTANCE",
        help="Include any atom that is within this distance (in real "
        "coordinates) of one of the specified atoms (via --atoms or "
        "--atoms-file). Distance is checked independently for each frame, "
        "so atoms can move in and out of this cutoff radius between frames. "
        "Default: disabled (only include specified atoms and their molecule)."
    )
    parser.add_argument(
        "--center", default=None,
        help="Atom to fix at the origin"
    )
    args = parser.parse_args()

    dump_path = Path(args.dump_file)
    if not dump_path.is_file():
        sys.exit(f"Error: file not found: {dump_path}")

    snapshot = args.snapshot
    if snapshot == "last":
        snapshot = ad.find_last_timestep(dump_path)
        if snapshot is None:
            sys.exit(f"Error: '{dump_path}' contains no snapshots.")

    out_path = Path(args.out_file)
    is_pdb = out_path.suffix.lower() == ".pdb"

    if is_pdb and snapshot is None:
        sys.exit(
            "Error: writing a PDB output file (extension '.pdb') requires "
            "-s/--snapshot to select a single snapshot to convert."
        )

    data_path = None
    if args.data_file is not None:
        data_path = Path(args.data_file)
        if not data_path.is_file():
            sys.exit(f"Error: file not found: {data_path}")
        if not is_pdb:
            print(
                f"Warning: --data-file '{data_path}' is ignored when writing "
                "an XYZ file (only PDB output supports CONECT records).",
                file=sys.stderr,
            )

    atoms = args.atoms
    exact_atoms = False
    if args.atoms_file:
        exact_atoms = True
        atoms = read_atoms_file(args.atoms_file)

    if is_pdb:
        bonds = parse_data_file_bonds(data_path) if data_path is not None else None
        n_frames = write_pdb(
            dump_path, args.out_file, atom_ids=atoms, exact_atoms=exact_atoms,
            target_timestep=snapshot, pbc=args.pbc, center=args.center, bonds=bonds,
            distance_cutoff=args.distance,
        )
    else:
        n_frames = write_xyz(
            dump_path, args.out_file, atom_ids=atoms, exact_atoms=exact_atoms,
            target_timestep=snapshot, pbc=args.pbc, center=args.center,
            distance_cutoff=args.distance,
        )
    print(f"Wrote {n_frames} frame(s) to '{args.out_file}'.")


if __name__ == "__main__":
    main()

