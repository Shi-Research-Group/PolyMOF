#!/usr/bin/python

from lammps_interface.lammps_main import LammpsSimulation
from lammps_interface.structure_data import from_CIF, write_CIF
import sys
from pathlib import Path

class Parameters:
    def __init__(self, cif):
        # File options
        self.cif_file = cif
        self.output_cif = False
        self.output_raspa = False

        # Force field options
        self.force_field = 'UFF'
        self.mol_ff = 'UFF'
        self.h_bonding = False
        self.dreid_bond_type = 'harmonic'
        self.fix_metal = False
        
        # Simulation options
        self.minimize = False
        self.bulk_moduli = False
        self.thermal_scaling = False
        self.npt = False
        self.nvt = False
        self.cutoff = 5
        self.replication = None
        self.orthogonalize = False
        self.random_vel = False
        self.dump_dcd = 0
        self.dump_xyz = 0
        self.dump_lammpstrj = 0
        self.restart = False
        
        # Parameter options
        self.tol = 0.4
        self.neighbour_size = 5
        self.iter_count = 10
        self.max_dev = 0.01
        self.temp = 298.0
        self.pressure = 1.0
        self.nprodstp = 200000
        self.neqstp = 200000
        
        # Molecule insertion options
        self.insert_molecule = ""
        self.deposit = 0

        self.thermal_anneal = False
        self.relax = False
        
    def show(self):
        for v in vars(self):
            print('%-15s: %s' % (v, getattr(self, v)))

if len(sys.argv) < 2:
    print("usage: go-lammps-interface file.cif [type:ufftype]")
    sys.exit(1)

infile = sys.argv[1]

if not Path(infile).exists():
    print("%s does not exist" % (infile))
    sys.exit(2)

par = Parameters(infile)

sim = LammpsSimulation(par)
cell, graph = from_CIF(par.cif_file)
if len(sys.argv) == 3:
    atomtype, ufftype = sys.argv[2].split(":")
    for n in graph.nodes:
        if graph.nodes[n]['element'] == atomtype:
            graph.nodes[n]['force_field_type'] = ufftype
sim.set_cell(cell)
sim.set_graph(graph)

sim.split_graph()
sim.assign_force_fields()
sim.compute_simulation_size()
sim.merge_graphs()

sim.write_lammps_files()
