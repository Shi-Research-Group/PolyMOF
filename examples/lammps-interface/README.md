# LAMMPS Interface

First a PDB is created with CONECT records for bonds between iron and nitrogen on various molecules ihcluding PBI and IM.  The files are converted to CIF format and LAMMPS Interface is invoked to create LAMMPS data and parameter files.  Since only the interactions involving Fe are desired, an additional script is used to extract those lines.  Finally, since the interaction parameters will be repeated in many of the files, the unique lines are extracted from all files.
