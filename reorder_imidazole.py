#!/usr/bin/env python
"""Find imidazole-type 5-rings (2 N, one bonded to H, one not) in a PDB file
with CONECT records and reorder atoms so each ring block reads:
    C (between the N's), N-H, C, C, N (no H)
The ring block is placed where the ring's first atom originally appeared.
Serials are renumbered and CONECT records rewritten.

Usage: reorder_imidazole.py in.pdb out.pdb
"""
import sys
from collections import defaultdict


def element(line):
    e = line[76:78].strip()
    if not e:
        e = line[12:16].strip().lstrip("0123456789")[:1]
    return e.capitalize()


def find_rings(adj, elem):
    """All simple 5-membered rings (as ordered atom cycles) among heavy atoms."""
    heavy = {i for i in adj if elem[i] != "H"}
    rings, seen = [], set()
    for a in heavy:
        stack = [[a]]
        while stack:
            path = stack.pop()
            if len(path) == 5:
                if a in adj[path[-1]]:
                    key = frozenset(path)
                    if key not in seen:
                        seen.add(key)
                        rings.append(path)
                continue
            for nb in adj[path[-1]]:
                if nb in heavy and nb not in path:
                    stack.append(path + [nb])
    return rings


# inp and out can be the same; file will be overwritten
def reorder_imidazole(inp, out):
    lines = open(inp).read().splitlines()
    atoms = {}   # serial -> line
    order = []
    conect = defaultdict(set)
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            s = int(l[6:11])
            atoms[s] = l
            order.append(s)
        elif l.startswith("CONECT"):
            f = [int(l[i:i + 5]) for i in range(6, len(l.rstrip()), 5)]
            for b in f[1:]:
                conect[f[0]].add(b)
                conect[b].add(f[0])
    elem = {s: element(l) for s, l in atoms.items()}
    adj = {s: conect[s] & set(atoms) for s in atoms}

    blocks = {}  # first original index -> ordered ring
    used = set()
    for ring in find_rings(adj, elem):
        ns = [a for a in ring if elem[a] == "N"]
        if len(ns) != 2 or used & set(ring):
            continue
        nh = [n for n in ns if any(elem[x] == "H" for x in adj[n])]
        if len(nh) != 1:
            continue
        nh = nh[0]
        n0 = ns[0] if ns[1] == nh else ns[1]
        c2 = [a for a in ring if nh in adj[a] and n0 in adj[a]]
        if len(c2) != 1:
            continue
        c2 = c2[0]
        # walk c2 -> NH -> C -> C -> N
        seq = [c2, nh]
        while len(seq) < 5:
            nxt = [x for x in adj[seq[-1]] if x in ring and x not in seq]
            seq.append(nxt[0])
        if seq[-1] != n0:
            continue
        used |= set(ring)
        blocks[min(order.index(a) for a in ring)] = seq

    new_order = []
    for i, s in enumerate(order):
        if i in blocks:
            new_order += blocks[i]
        elif s not in used:
            new_order.append(s)
    assert len(new_order) == len(order)
    newid = {s: i + 1 for i, s in enumerate(new_order)}

    res = []
    first = True
    for l in lines:
        if l.startswith(("ATOM", "HETATM")):
            if first:
                for s in new_order:
                    res.append(atoms[s][:6] + "%5d" % newid[s] + atoms[s][11:])
                first = False
        elif l.startswith("CONECT"):
            continue
        elif l.startswith("TER"):
            res.append("TER   %5d" % (len(order) + 1))
        else:
            res.append(l)
    # put CONECT before END
    cl = []
    for s in sorted(newid, key=newid.get):
        nb = sorted(newid[b] for b in adj[s])
        cl.append("CONECT%5d" % newid[s] + "".join("%5d" % b for b in nb))
    if res and res[-1].startswith("END"):
        res = res[:-1] + cl + [res[-1]]
    else:
        res += cl
    open(out, "w").write("\n".join(res) + "\n")
    print("Reordered %d ring(s)" % len(blocks))

def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    reorder_imidazole(*sys.argv[1:])

if __name__ == "__main__":
    main()
