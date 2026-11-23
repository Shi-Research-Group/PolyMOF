# antechamber

The `go` script builds 10mer files and then produces their prmtop files

```
python ../../polymof/extract_monomer.py --bond 154:99 --attach 133 --polymerize 10 pentamer.pdb 10mer.pdb
```

This instructs the script to define the monomer as being between atoms 154 and 133 (inclusive).  The bond between atoms 154 and 99 is broken and then a new bond formed between 154 and 133 for each repeat unit in the final polymer.  The script shifts the new repeat unit so that the new bond between 133 and new 154 is equal to the original bond vector between 99 and 154.  The process is illustrated below.

![Illustration of the polymerization process with atoms 154, 99 and 133 numbered.](pentamer-extract.png)

The polymerization process is repeated 10 times and then hydrogens attached to the atoms on the ends equivalent to the original 154 and 133 with the original bond vectors.

```
for h in 0467554 0527564 0568015 0619877 0984613 1045277; do
  python ../../polymof/flip_hydrogens.py 10mer.pdb 10mer${h}.pdb $h
done
```

This loops over six tautomers and flips the hydrogens between nitrogens to generate new PDB files.  Each bit in the number corresponds to one hydrogen; 0 indicates H is closer to the bridge between benzenes, 1 indicates H is farther away.

```
python ../../polymof/build_prmtop_from_pdb.py --old-prmtops 3mers/*.prmtop --new-pdbs 10mer*.pdb
```

This generates a prmtop file for each tautomer from a exhaustive library of prmtop files of all 64 three-mer tautomers.  Each atom in each decamer is matched against an atom or atoms in the three-mers that has the same local topology (out to atoms 10 bonds away) and the decamer atom's parameters are set to matching atoms from the 3mers.  When multiple matches are found, all matches must have the same atom types and bonding parameters and the new charge for the decamer is taken to be the average across all matches.  The resulting decamer will have a small net charge, so the charges on all atoms are shifted by an amount that makes the molecule have zero net charge.
