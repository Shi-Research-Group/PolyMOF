#!/usr/bin/env python

import sys
import argparse
import reaction_rate
import numpy as np
from scipy.optimize import curve_fit
import math

def setup_args(parser):
  parser.add_argument("--sim", required=True, help="Simulation result for metal bond population")

def parse_args():
  parser = argparse.ArgumentParser(
          description="Fit rate constants for simple reaction theory")
  reaction_rate.setup_args(parser)
  setup_args(parser)
  args = parser.parse_args()
  return args

class ReactionSolver:
    def __init__(self, sites, c0, cN, dt, nt, allT, nsim, y):
        self.sites = sites
        self.c0 = c0
        self.cN = cN
        self.dt = dt
        self.nt = nt
        self.allT = allT
        self.nsim = nsim
        self.y = y

    def reaction(self, it, *k):
        i, t = it
        k = np.asarray([math.exp(k[i]) for i in range(len(k))])
        result = reaction_rate.computeReaction(self.sites, self.c0, self.cN, k, self.dt, self.nt)
        tmap = {}
        rv = np.zeros(i.size)
        sumdiff2 = np.zeros(self.sites+1)
        sd2 = 0
        for j in range(i.size):
            ij = int(i[j])
            tj = t[j]
            jj = int(round(tj / self.dt))
            if jj >= len(result):
                print("Not enough timesteps.  sim data goes to ", tj, " but model only goes to ", self.nt*self.dt)
                sys.exit(1)
            rv[j] = result[jj][ij+1]   # result has cN at the beginning

            # now check against sim data
            d = rv[j] - self.y[j]
            sumdiff2[ij] += d*d
            sd2 += d*d
        for a in range(len(k)):
            avgerr = math.sqrt(sumdiff2[a]/i.size*(self.sites+1))
            print(f"{k[a]:.3e} {avgerr:.3e}  ", end="")
        avgerr = math.sqrt(sd2/i.size)
        print(f"{avgerr:.3e}")
        return rv
        
def main():
    args = parse_args()
    
    nsim = []
    allT = []
    with open(args.sim, "r") as f:
        for l in f:
            bits = l.split()
            allT.append(float(bits[0]))
            nsim.append([float(bits[i]) for i in range(1,len(bits))])

    tLastSim = allT[len(allT)-1]
    args.nt = int(round(tLastSim/args.dt))
    alli = [i for i in range(args.sites+1)]
    allT = np.asarray(allT)
    alli = np.asarray(alli)
    allimg, allTmg = np.meshgrid(alli, allT)
    x = np.vstack((allimg.ravel(), allTmg.ravel()))
    y = np.asarray(nsim)
    y = y.ravel()

    if len(nsim) == 0:
        sys.exit("No simulation data")
    if len(nsim[0]) != args.sites + 1:
        print("Sim data columns", len(nsim[0]), "but, but --sites", args.sites)
        sys.exit(1)
    if nsim[0][0] != args.metal:
        print("Sim data has", nsim[0][0], "metals, but --metal ", args.metal)
        sys.exit(1)

    reaction = ReactionSolver(args.sites, args.metal, args.nitrogen, args.dt, args.nt, allT, nsim, y)
    k_guess = np.log(np.asarray(args.k))
    k_min = np.asarray([-45 for i in range(args.sites)])
    k_max = np.asarray([0 for i in range(args.sites)])
    params_opt, params_cov = curve_fit(reaction.reaction, x, y, k_guess, bounds=(k_min,k_max))
    k_fit = np.exp(params_opt)
    print("fit: ", " ".join([str(k_fit[i]) for i in range(len(k_fit))]))


if __name__ == "__main__":
    main()
