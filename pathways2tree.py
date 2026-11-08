#!/usr/bin/env python3
"""
pathways2tree.py

Reads a --pathways file where each line is:

    <label> <count>

The label's length determines its depth in a binary tree: the root is the
first line (a single-character label). Every other label's last character
is the branch taken from its parent (whose label is the current label minus
its last character). The count (2nd column) is drawn as the label of the
edge connecting a node to its parent. A count of 0 means the node (and,
naturally, everything below it) is omitted from the diagram.

Labels that are anagrams of each other (contain the same letters, just in a
different order, e.g. "ZAC" and "ZCA") represent the same underlying state
reached via different reaction orders, so they are combined into a single
node in the diagram. The alphabetically-first label is used as the combined
node's name, and it keeps all of the incoming edges (with their original
counts) and all of the children from every original label that was merged
into it.

The tree is rendered to an image using Graphviz's `dot`.

Usage:
    ./pathways2tree.py --pathways reaction.pathways --output reaction_tree.png
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_pathways(path):
    """Return dict: label -> count, preserving file order."""
    nodes = {}
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected 2 columns, got {len(parts)!r}: {raw!r}"
                )
            label, count_str = parts
            try:
                count = int(count_str)
            except ValueError:
                raise ValueError(f"{path}:{lineno}: count is not an integer: {count_str!r}")
            nodes[label] = count
    return nodes


def build_graph(nodes):
    """
    Merge labels that are anagrams of each other (same letters, different
    order) into a single combined node/group, then determine which groups
    are reachable from the root: a group is reachable if the root, or if
    at least one of its member labels has a non-zero count *and* its
    literal (string-prefix) parent's group is itself reachable. This is
    evaluated at the group level (not the raw label level) so that, e.g.,
    "ZCBA" (count 0) is still considered reachable if its anagram "ZCAB"
    (count 2) has a valid, reachable parent chain -- a child listed under
    the zero-count spelling (like "ZCBAC") is then attached to that
    reachable merged node instead of being dropped.

    Returns (root_id, node_ids, edges) where:
      - root_id is the combined-node id for the root.
      - node_ids is the sorted set of reachable combined-node ids to draw.
      - edges is a sorted list of (parent_id, child_id, edge_value) triples,
        one per surviving original label (a combined node may thus receive
        more than one incoming edge, one per original parent it came from).
    """
    if not nodes:
        raise ValueError("pathways file contained no nodes")

    root_label = next(iter(nodes))

    # Group ALL labels (regardless of count) by their multiset of
    # characters (anagrams), and pick the alphabetically first label in
    # each group as its name.
    groups = {}
    for label in nodes:
        key = tuple(sorted(label))
        groups.setdefault(key, []).append(label)

    canon_of = {}
    for labels in groups.values():
        rep = min(labels)
        for label in labels:
            canon_of[label] = rep

    root_id = canon_of[root_label]

    # Process labels in increasing length order so that a group's
    # reachability (established via any one of its members) is always
    # known before its children are considered.
    reachable_ids = {root_id}
    edges = set()
    for label in sorted(nodes, key=len):
        if label == root_label:
            continue
        count = nodes[label]
        if count == 0:
            continue
        parent_label = label[:-1]
        parent_id = canon_of.get(parent_label)
        if parent_id is None or parent_id not in reachable_ids:
            continue
        child_id = canon_of[label]
        reachable_ids.add(child_id)
        edges.add((parent_id, child_id, count))

    node_ids = sorted(reachable_ids)
    return root_id, node_ids, sorted(edges)


def to_dot(root_id, node_ids, edges):
    lines = [
        "digraph pathways {",
        '  rankdir=TB;',
        '  node [shape=box, style=rounded, fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]

    def node_id(label):
        return f'n_{label}'

    for label in node_ids:
        display = label if label else "(root)"
        lines.append(f'  {node_id(label)} [label="{display}"];')

    for parent_label, child_label, edge_value in edges:
        lines.append(
            f'  {node_id(parent_label)} -> {node_id(child_label)} [label="{edge_value}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def check_pathways(nodes):
    """
    Validate that, for every node with children, the sum of the children's
    counts does not exceed the node's own count. Returns True if all nodes
    are consistent, False if any violation was found (details are printed).
    """
    children = {label: [] for label in nodes}
    for label in nodes:
        if len(label) <= 1:
            continue
        parent_label = label[:-1]
        if parent_label in children:
            children[parent_label].append(label)

    for label in children:
        children[label].sort()

    ok = True
    for label, child_labels in children.items():
        if not child_labels:
            continue
        count = nodes[label]
        child_sum = sum(nodes[c] for c in child_labels)
        if child_sum > count:
            ok = False
            print(f"Node {label!r} (n={count}) has children summing to {child_sum} > {count}:")
            for c in child_labels:
                print(f"    {c!r}: n={nodes[c]}")

    if ok:
        print("OK: all nodes' children sum to no more than the node's own count.")
    return ok


def render(dot_source, output_path):
    fmt = Path(output_path).suffix.lstrip(".") or "png"
    try:
        subprocess.run(
            ["dot", f"-T{fmt}", "-o", str(output_path)],
            input=dot_source,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        sys.exit("error: Graphviz 'dot' executable not found; please install graphviz.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: 'dot' failed with exit code {e.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Render a pathways file as a binary tree image.")
    parser.add_argument("pathways", help="Path to the pathways input file.")
    parser.add_argument(
        "--output",
        help="Output image path (extension selects the format, e.g. .png, .svg, .pdf).",
    )
    parser.add_argument(
        "--dot-out",
        default=None,
        help="Optional path to also save the intermediate Graphviz .dot source.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Instead of drawing an image, check that each node's children's "
        "counts sum to no more than the node's own count, and report violations.",
    )
    args = parser.parse_args()

    nodes = parse_pathways(args.pathways)

    if args.check:
        ok = check_pathways(nodes)
        sys.exit(0 if ok else 1)

    root_id, node_ids, edges = build_graph(nodes)
    dot_source = to_dot(root_id, node_ids, edges)

    if args.dot_out:
        Path(args.dot_out).write_text(dot_source)

    if not args.output:
        args.output = args.pathways + ".png"

    render(dot_source, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
