#!/usr/bin/env python

import sys
import os
import argparse
import re
from grablammps import readLog

def parse_args():
  def list_of_maps(arg):
      return list(arg.split(","))
  parser = argparse.ArgumentParser(
      description="Count bonds of various types from reaction counters")
  parser.add_argument("reactions_files", nargs='+', help="File that contains list of reactions")
  parser.add_argument("--map", type=list_of_maps, default=[])
  parser.add_argument("--log", required=True, dest='logs', nargs='+')
  parser.add_argument("--pathways", action='store_true', help="Only print the first section of output with reaction pathway counts")
  parser.add_argument("--nvst", action='store_true', help="Print the metal bond population vs time")
  parser.add_argument("--centers", type=int, help="Total number of reaction centers")
  parser.add_argument("-n")
  args = parser.parse_args()
  if args.nvst:
    if not args.centers:
      print("You must specify centers if you want n vs. t")
      sys.exit(1)
    if not args.n:
      args.n = "all"
  if args.pathways:
    if not args.n:
      args.n = "all"
  for j in range(len(args.map)):
    bits = args.map[j].split(":")
    if len(bits) != 2 or len(bits[0]) != 1 or len(bits[1]) != 1 or bits[0] == bits[1]:
      parser.error("invalid map " + args.map[j])
  if len(args.reactions_files) != len(args.logs):
    parser.error("Number of reactions files ("+str(len(args.reactions_files))+") must match number of log files ("+str(len(args.logs))+")")
  for rf in args.reactions_files:
    if not os.path.exists(rf):
      parser.error("Reaction file "+rf+" must exist")
  for lf in args.logs:
    if not os.path.exists(lf):
      parser.error("Log file "+lf+" must exist")
  return args


def parse_reactions(reactions_file, map):
  """Returns a list of all reactions"""
  allReactions = []

  with open(reactions_file, "r") as f:
    go = 0
    refix = re.compile("^fix .* bond/react")
    for l in f:
      l = l.split("#", 1)[0].strip()
      if not l:
        continue
      if l == "Reactions":
        # raw list of reactions
        go = 1
        continue
      elif refix.match(l):
        # LAMMPS fix bond/react
        go = 2
        continue
      if go == 0:
        continue
      bits = l.split()
      if go == 1:
        r = bits[0]
      else:
        r = bits[1].split("_")[1]
      for j in range(len(map)):
        bits = map[j].split(":")
        r = r.replace(bits[0], bits[1])
      n = len(r)-1
      while len(allReactions) <= n:
        allReactions.append([])
      allReactions[n].append(r)
  if len(allReactions) == 0:
    print("Could not find any reactions")
    sys.exit(1)
  return allReactions

def getNumReactants(nBonds, allReactions, nvst):

  nReactants = None

  if not nBonds:
    nReactants = len(allReactions)
  elif re.compile("\\d+").match(nBonds):
    if nvst:
      print("Cannot get n vs. t while specifying n")
      sys.exit(1)
    nReactants = int(nBonds) + 1
    if nReactants < 2 or nReactants > len(allReactions):
      print("n must be at least 1 and at most ", len(allReactions)+1)
      sys.exit(1)
  elif nBonds == "all":
    nReactants = "all"
  else:
    sys.exit("Unrecognized n; must be number of bonds or 'all'")
  return nReactants

def getStartStop(nReactants, allReactions):
  # start and stop are 0-based indices into the the list of all reactions
  if nReactants == "all":
    totalReactions = sum([len(ar) for ar in allReactions])
    start = 0
    stop = totalReactions - 1
    newar = []
    for ar in allReactions:
      newar += ar
    allReactions = newar
  else:
    start = sum([len(allReactions[i]) for i in range(nReactants-1)])
    stop = start + len(allReactions[nReactants-1]) - 1
    allReactions = allReactions[nReactants-1]
  return start, stop, allReactions

def getAllC(allReactions):
  myReactions = [(len(r), r) for r in allReactions]
  myReactions.sort()
  myReactions = [r for nReactants,r in myReactions]
  allC = {}
  for r in myReactions:
    for i in range(1,len(r)):
      allC[r[i]] = True
  allC = list(allC.keys())
  allC.sort()
  return allC

def getReactionRange(log, nReactants, start, stop, nvst):
  headers = readLog(log, None, True)
  reactionRange = ""
  if nReactants == "all":
    for h in headers:
      for hh in h:
        if hh[:6] == "f_rxns":
          reactionRange = hh+"-"+h[-1]
          break
      if reactionRange:
        break
    reactionRange = [reactionRange]
  else:
    reactionRange = ["f_rxns["+str(start+1)+"]-f_rxns["+str(stop+1)+"]"]
  if nvst:
    reactionRange = ["Step"] + reactionRange
  return reactionRange

def analyze_log_row(row, iStart, iStop, iAllReactions, allC, nvst):
  reactionsCount = dict.fromkeys(iAllReactions, 0)
  expectedColumns = len(iAllReactions)
  productCounts = {}
  totalCounts = dict.fromkeys(allC, 0)
  if len(row) != expectedColumns:
    print("I have", expectedColumns, "reactions and", len(row), "columns")
    sys.exit(2)
  for i in range(iStart, iStop+1):
    nr = int(row[i-iStart])
    r = iAllReactions[i-iStart]
    reactionsCount[r] += nr
    productKey = ""
    reactantKey = ""
    for c in allC:
      productKey += str(r.count(c))+c
      reactantKey += str(r[:-1].count(c))+c

    # this only works if user specified n=all
    totalCounts[r[len(r)-1]] += nr
    if not productKey in productCounts:
      productCounts[productKey] = 0
    if reactantKey not in productCounts:
      productCounts[reactantKey] = 0
    productCounts[productKey] += nr
    productCounts[reactantKey] -= nr
  return productCounts, totalCounts, reactionsCount

def extractNVST(data, start, stop, allReactions, allC, centers, startStep, startCounts):
  # our first column will be Step
  start += 1
  stop += 1
  step = None
  step0 = None
  nCounts = []
  for row in data:
    if step0 == None:
      step0 = int(row[0])
      if step0 == 0 and startStep > 0:
        continue
    productCounts, totalCounts, reactionsCount = analyze_log_row(row[1:], start, stop, allReactions, allC, True)
    n = len(allReactions[len(allReactions)-1])
    nCounts = startCounts.copy()
    nCounts[1] = centers
    for r in reactionsCount:
      nCounts[len(r)] += reactionsCount[r]
    nCounts2 = [nCounts[i]-nCounts[i+1] for i in range(1,len(nCounts)-1)]
    nCounts2.append(nCounts[len(nCounts)-1])
    step = startStep + int(row[0]) - step0
    print(str(step)+" "+(" ".join([str(nc) for nc in nCounts2])))
  return step, nCounts

def addToDict(sumDict, newDict):
  for k, v in newDict.items():
    if k not in sumDict:
      sumDict[k] = 0
    sumDict[k] += v

def main():
  args = parse_args()
  count_bonds(args.logs, args.reactions_files, args.map, args.pathways, args.nvst, args.n, args.centers)

def count_bonds(logs, reactions_files, reactant_map, pathways, nvst, n, centers):
  productCounts = {}
  totalCounts = {}
  reactionsCounts = {}
  for i in range(len(logs)):
    log = logs[i]
    reactions_file = reactions_files[i]
    iAllReactions = parse_reactions(reactions_file, reactant_map)
    if i == 0:
      if not pathways and not nvst:
        print(str(sum([len(iAllReactions[i]) for i in range(len(iAllReactions))]))+" reactions")

      nReactants = getNumReactants(n, iAllReactions, nvst)

    iStart, iStop, iAllReactions = getStartStop(nReactants, iAllReactions)

    if i == 0:
      allReactions = iAllReactions

      start = iStart
      stop = iStop

      allC = getAllC(iAllReactions)
      n = len(allReactions[len(allReactions)-1])
      startStep = 0
      startCounts = [0 for i in range(n+1)]

    reactionRange = getReactionRange(log, nReactants, iStart, iStop, nvst)

    data = readLog(log, reactionRange, False)

    if nvst:
      startStep, startCounts = extractNVST(data, iStart, iStop, iAllReactions, allC, centers, startStep, startCounts)
      continue

    iProductCounts, iTotalCounts, iReactionsCounts = analyze_log_row(data[-1], iStart, iStop, iAllReactions, allC, False)
    addToDict(productCounts, iProductCounts)
    addToDict(totalCounts, iTotalCounts)
    addToDict(reactionsCounts, iReactionsCounts)

  if nvst:
    return

  if not pathways:
    print("Reaction counts")
  for r,c in reactionsCounts.items():
    print(r, c)
  if pathways:
    return
  print("")
  pks = list(productCounts.keys())
  pks.sort()
  print("Product counts")
  for pk in pks:
    # if n=all, then the product corresponding to the original metal will have negative counts
    # if n is a number, then products corresponding to the previous level reactants will have negative counts
    if productCounts[pk] > 0:
      print(pk, productCounts[pk])
  print("")
  if n == "all":
    print("Total counts")
    for c in allC:
      print(c, totalCounts[c])

if __name__ == "__main__":
    main()
