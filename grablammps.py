#!/usr/bin/env python

import sys
import re
import os
import argparse

def parse_args():
  parser = argparse.ArgumentParser(prog="grablammps",
                                   description="Grabs data from LAMMPS log file")

  parser.add_argument("--log", help='log file to read', required=True)
  parser.add_argument("--headers", action='store_true', help='just print the headers from the log file')
  parser.add_argument("columns", nargs='*',
                      help='multiple columns may be given or a range of columns')

  args = parser.parse_args()

  if not args.headers and len(args.columns) == 0:
    print("You must specify either columns or --headers")
    sys.exit(0)

  if not os.path.exists(args.log):
    print("Cannot find log file ", args.log)
    sys.exit(1)

  cols = args.columns
  if len(cols) == 1:
    args.columns = ["Step", cols[0]]

  return args

def readLog(fn, cols, justHeaders):
  icols = []
  printNow = False
  nheaders = -1
  allHeaders = {}
  data = []
  with open(fn, "r", encoding="utf-8") as f:
    Lto = re.compile("Loop time of")
    prevLine = None
    prevLineError = None
    for l in f:
      if prevLineError:
        print(prevLineError)
        print(prevLine)
        sys.exit(1)
      bits = l.split()
      if len(bits) == 0:
        continue
      if len(icols) == 0 and bits[0] == "Step":
        if justHeaders:
          if not l in allHeaders:
            data.append(l.strip().split())
            allHeaders[l] = True
          continue
        printNow = True
        nheaders = len(bits)
        for ic in range(0, len(cols)):
          icmin=cols[ic]
          icmax=cols[ic]
          if "-" in cols[ic]:
            cbits = cols[ic].split("-")
            if len(cbits) != 2:
              print("column specification has more than one -")
              sys.exit(1)
            icmin=cbits[0]
            icmax=cbits[1]
          found = False
          for i in range(0, len(bits)):
            if bits[i] == icmin:
              found = True
              foundMax = False
              for j in range(i, len(bits)):
                icols.append(j)
                if bits[j] == icmax:
                  foundMax = True
                  break
              if not foundMax:
                print("Found", icmin, " but could not find max column in range ", icmax)
          if not found:
            #print("Could not find column %s" % (cols[ic]))
            #print("Header: %s" % (l), end='')
            #sys.exit(2)
            printNow = False
            nheaders = -1
            icols = []
            break
        continue
      if printNow:
        if Lto.match(l) == None and bits[0] != "WARNING:":
          if len(bits) != nheaders:
            # If the sim is still running, the last line may be an incomplete
            # thermo line.  If so, just print a warning.  If the unrecognized
            # line is not the last one, then print and exit.
            prevLine = l
            prevLineError = "Expected %d columns, but found %d" % (nheaders, len(bits))
            continue
          row = [bits[i] for i in icols]
          data.append(row)
        else:
          printNow = False
          icols = []
  if prevLineError:
    print("Log file incomplete")
  return data

def main():
  args = parse_args()
  data = readLog(args.log, args.columns, args.headers)
  for row in data:
    for c in row:
      print("%s " % c, end='')
    print("")

if __name__ == "__main__":
    main()
