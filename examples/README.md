# PolyMOF examples

These examples illustrate how the scripts in the `polymof` directory are used.  They are representative of the actions needed for the various stages of the workflow.  The `collect.py` script can be used to drive most of this process, but may not be appropriate for all workflows.

## antechamber

While antechamber can determine GAFF2 parameters and charges for small molecules including imidazole (IM) and nitrate that are part of the simulations, it struggles with even short polymers.  This directory contains input files and a script to generate GAFF2 parameters for PBI decamers (several tautomers).

First, the script extracts a monomer from a PBI pentamer and constructs the decamer.  Then the various tautomers are generated based on a bitmask that specifies the placement of hydrogens along the backbone chain.  Lastly, the GAFF2 parameters and charges are determined for all atoms in the tautomers by matching the atom's local topology against atoms in a library of three-mer PRMTOP that had been previously generated with antechamber.

## LAMMPS Interface

The LAMMPS simulation will use a hybrid forcefield consisting of GAFF2 parameters for PBI and IM, but UFF parameters for any interaction involving Fe.  The scripts in this directory determine those UFF parameters given PDB input files that describe the molecules.

Generates UFF interaction parameters are needed for Fe complexes using [LAMMPS Interface](https://github.com/peteboyd/lammps_interface).  The example script does this for one complex, but the complete workflow would need to repeat this for enough complexes to cover all the possible interactions.  The end result is a `extra.param` file that contains all the UFF parameters (in LAMMPS format) involving iron.  The scripts that set up the reaction templates will need this file to construct the post-reaction templates.

## reaction

The script in this directory sets up and runs a short LAMMPS simulation as specified by a `params` file that specifies properties like PBI chain length and system composition.  Many of the other files would be generated in previous steps, but were added to allow the example script to execute independent of those steps.

The script generates LAMMPS data and parameter files using [AMBER2LAMMPS](https://github.com/askforarun/AMBER2LAMMPS) and then generates the pre- and post-reaction templates along with corresponding maps and LAMMPS commands to include in the input script.  Finally, the LAMMPS reaction simulation is run.
