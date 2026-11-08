# PolyMOF reaction scripts

This is a collection of scripts used to setup, manage and analyze PolyMOF simulations in LAMMPS.

## Initial Setup

These scripts help to set up the initial reaction simulation, including antechamber, LAMMPS Interface, PACKMOL, and reaction templates.

* `extract_monomer.py` extracts and replicates a monomer to form an oligomer of a given size.
* `build_prmtop_from_pdb.py` creates a `prmtop` file for a given tautomer pdb file from `prmtop` files for other tautomers.
* `cif_set_atom_type.py` sets iron to be a particular UFF type in cif file for LAMMPS Interface.
* `combine_pdbs.py` combines two or more PDB files and adds CONECT records.
* `data2molecule.py` creates a LAMMPS molecule template from a LAMMPS data file.
* `data2reaction.py` creates LAMMPS pre- and post-reaction molecule templates and a reaction map from LAMMPS data files.
* `extract_data.py` extracts interactions involving iron from data file produced by LAMMPS Interface.
* `find_unique_fragments.py` identifies all unique reactive fragments in a data file.
* `flip_hydrogens.py` flip hydrogens between nitrogens to create a particular tautomer.
* `get_mass.py` computes mass for a pdb file.
* `go_lmp_interface.py` wrapper script for LAMMPS Interface.
* `grablammps.py` extracts data from a LAMMPS log file.
* `insert_molecules.py` adds (nitrate) molecules to a LAMMPS system.
* `make_pentamer_ac.py` sets up PBI antechamber runs.
* `make_reactions.py` creates a list of reactions for a given set of reactants.
* `polymof_setup.py` utility functions that require extra imports 
* `polymof_util.py` utility functions that don't require extra imports
* `react_molecule.py` produces a post-reaction template from a pre-reaction template.
* `reorder_imidazole.py` reorders atoms within an imidazole ring (isolated or in PBI) so that antechamber identifies all bare nitrogens as `nd`.

## Management

These scripts help to manage and monitor the progress of simulations as well as setup up additional simulations after the initial one.

* `alter_charges.py` redistribute some of Fe+3 charge to bonded nitrogens
* `collect.py` monitor simulations across all systems and set up new sims.
* `continue_reaction.py` set up a new reaction simulation to continue an old one.
* `prepare_anneal.py` prepare an annealing simulation to run from the final configuration of a reaction simulation.
* `prune_reactions.py` remove reactions that can no longer happen.
* `remove_molecules.py` remove molecules (thing not bonded to PBI) from a data file.
* `replace_params.py` switch some irons from Fe6 (octahedral) to Fe3 (tetrahedral)

## Analysis

These scripts analyze the output of simulations.

* `analyze_complexes.py` identify complexes and provide some basic analysis of their structure.
* `analyze_dump.py` analyze a LAMMPS dump file in various ways.
* `count_bonds.py` count bonds formed from a LAMMPS log file.
* `dump2xyz.py` extract an xyz file from a LAMMPS dump.
* `pathways2tree.py` construct a visual representation of a reaction pathway.
* `plot_molecule_graph.py` construct a visual representation of a molecule template or pre- and post-reaction templates.

## Gaussian

These scripts handle Gaussian runs of Fe-IMx clusters.  Each cluster does geometry optimization and then energy calculation for B3LYP and then MP2.

* `gcollect.py` setup and monitor Gaussian runs and antechamber runs.
* `extract_charge.py` extract charges from .mol file.
* `gextractopt.py` extract optimized geometry from Gaussian log file.
