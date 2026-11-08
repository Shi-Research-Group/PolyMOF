#!/usr/bin/env python

import sys
import argparse

def setup_args(parser):
  def list_of_k(arg):
      return list([float(x) for x in arg.split(",")])

  parser.add_argument("--sites", required=True, type=int, help="Number of sites per metal")
  parser.add_argument("--metal", required=True, type=float, help="Initial concentration of metal")
  parser.add_argument("-N", "--nitrogen", required=True, type=float, help="Initial concentration of nitrogen")
  parser.add_argument("-k", required=True, type=list_of_k, help="Rate constant")
  parser.add_argument("--dt", required=True, type=float, help="Timestep")
  parser.add_argument("--nt", required=True, type=int, help="Number of timesteps")

def parse_args():
  parser = argparse.ArgumentParser(
          description="Compute reaction progress from simple reaction theory")
  setup_args(parser)
  args = parser.parse_args()
  return args

def computeReaction(sites, c0, cN, k, dt, nt):
    c = [0 for i in range(sites+1)]
    c[0] = c0
    t = 0
    cAll = [[cN, *c]]

    for i in range(nt):
        cNext = [c[i] for i in range(len(c))]
        for j in range(len(c)):
            if j < sites:
                cNext[j] -= k[j] * c[j] * cN * dt
            if j > 0:
                cNext[j] += k[j-1] * c[j-1] * cN * dt
            cNext[j] = max(cNext[j], 0)
            cNext[j] = min(cNext[j], c0)
        cNnext = cN - sum([c[j]*k[j] for j in range(len(c)-1)]) * cN * dt
        c = cNext
        cN = cNnext
        cAll.append([cN, *c])
    return cAll


def main():
    args = parse_args()
    cAll = computeReaction(args.sites, args.metal, args.nitrogen, args.k, args.dt, args.nt)
    for i in range(len(cAll)):
        print(i*args.dt, " ".join([str(cAll[i][j]) for j in range(len(cAll[i]))]))

if __name__ == "__main__":
    main()
