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
        # Display name: first character kept, remaining characters sorted
        rep = rep[0] + "".join(sorted(rep[1:]))
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


NAMED_COLORS = {
    "red": (255, 0, 0), "green": (0, 160, 0), "blue": (0, 0, 255),
    "yellow": (255, 255, 0), "orange": (255, 165, 0), "purple": (128, 0, 128),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "pink": (255, 192, 203),
    "brown": (165, 42, 42), "gray": (128, 128, 128), "grey": (128, 128, 128),
    "black": (0, 0, 0), "white": (255, 255, 255), "gold": (255, 215, 0),
    "lightblue": (173, 216, 230), "lightgreen": (144, 238, 144),
    "salmon": (250, 128, 114), "teal": (0, 128, 128),
}


def parse_color(text):
    """Parse '#rrggbb' or a name from NAMED_COLORS into an (r, g, b) tuple."""
    t = text.strip().lower()
    if t.startswith("#") and len(t) == 7:
        try:
            return tuple(int(t[i:i + 2], 16) for i in (1, 3, 5))
        except ValueError:
            pass
    elif t in NAMED_COLORS:
        return NAMED_COLORS[t]
    raise ValueError(
        f"unknown color {text!r}: use #rrggbb or one of: {', '.join(sorted(NAMED_COLORS))}"
    )


def parse_letter_colors(specs):
    """Turn ['I=red', 'P=#00ff00'] into {'I': (r, g, b), 'P': (r, g, b)}."""
    colors = {}
    for spec in specs or []:
        letter, sep, color = spec.partition("=")
        if not sep or len(letter) != 1:
            raise ValueError(f"--color expects LETTER=COLOR (e.g. I=red), got {spec!r}")
        colors[letter] = parse_color(color)
    return colors


def node_fill(label, letter_colors):
    """
    Blend the colors of the letters in the label (excluding the first,
    root, character), weighted by how often each appears. Returns
    (fill_hex, font_hex) or None if no letter in the label has a color.
    """
    parts = [letter_colors[c] for c in label[1:] if c in letter_colors]
    if not parts:
        return None
    r, g, b = (sum(p[i] for p in parts) / len(parts) for i in range(3))
    # Pick black or white text depending on fill brightness
    font = "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "#ffffff"
    return f"#{round(r):02x}{round(g):02x}{round(b):02x}", font


def to_dot(root_id, node_ids, edges, max_width=8.0, letter_colors=None,
           numbers="none", root_total=0, positions=None, edge_numbers=True):
    letter_colors = letter_colors or {}
    lines = [
        "digraph pathways {",
        '  rankdir=TB;',
        '  splines=line;' if positions else '',
        '  node [shape=box, style=rounded, fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]

    def node_id(label):
        return f'n_{label}'

    # entered = total arriving at a node (the root's own count for the root);
    # terminated = entered minus everything that went on to children.
    entered = {label: 0 for label in node_ids}
    entered[root_id] = root_total
    left = {label: 0 for label in node_ids}
    for parent_label, child_label, edge_value in edges:
        entered[child_label] += edge_value
        left[parent_label] += edge_value

    for label in node_ids:
        display = label if label else "(root)"
        attrs = f'label="{display}"'
        if numbers == "entered":
            attrs = f'label="{display}\\n{entered[label]}"'
        elif numbers == "terminated":
            term = entered[label] - left[label]
            if term and root_total:
                pct = 100.0 * term / root_total
                attrs = f'label="{display}\\n{pct:.1f}%"'
        fill = node_fill(label, letter_colors)
        if fill:
            attrs += f', style="rounded,filled", fillcolor="{fill[0]}", fontcolor="{fill[1]}"'
        if positions:
            attrs += f', pos="{positions[label][0]:.1f},{positions[label][1]:.1f}"'
        lines.append(f'  {node_id(label)} [{attrs}];')

    # Edge thickness is proportional to count (largest edge = max_width),
    # with a small floor so very small edges stay visible.
    values = [v for _, _, v in edges]
    top = max(values) if values else 0

    def penwidth(value):
        if top == 0:
            return 1.0
        return max(0.5, max_width * value / top)

    placed = []  # label positions already used, to keep numbers apart

    for i, (parent_label, child_label, edge_value) in enumerate(edges):
        attrs = f'penwidth={penwidth(edge_value):.2f}'
        if not edge_numbers:
            pass
        elif positions:
            # neato -n ignores edge label positions, so draw the number as a
            # borderless node beside the edge midpoint. Try both sides and keep
            # the one farther from numbers already placed (ties: outer side).
            (x1, y1), (x2, y2) = positions[parent_label], positions[child_label]
            offset = 14 + penwidth(edge_value)
            # Shift toward the parent along the edge so the number is centered
            # on the visible shaft (the arrowhead takes up the child's end)
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            shift = 6.0
            preferred = -1 if x2 < x1 else 1
            best = None
            for side in (preferred, -preferred):
                lx = (x1 + x2) / 2 + side * offset + shift * (x1 - x2) / length
                ly = (y1 + y2) / 2 + shift * (y1 - y2) / length
                clearance = min(
                    (((lx - px) ** 2 + (ly - py) ** 2) ** 0.5 for px, py in placed),
                    default=float("inf"),
                )
                # require a clear margin before giving up the preferred side
                if best is None or clearance > best[0] + 1e-9 and best[0] < 28:
                    best = (clearance, lx, ly)
            _, lx, ly = best
            placed.append((lx, ly))
            lines.append(
                f'  e{i} [shape=plaintext, label="{edge_value}", width=0, height=0, '
                f'margin=0.02, pos="{lx:.1f},{ly:.1f}"];'
            )
        else:
            attrs += f', label="{edge_value}"'
        lines.append(
            f'  {node_id(parent_label)} -> {node_id(child_label)} [{attrs}];'
        )

    lines.append("}")
    return "\n".join(lines)


def compute_positions(node_ids, dx=82.5, dy=75.0):
    """
    Place nodes on a grid so every edge is exactly diagonal: depth sets the
    vertical position and the letters set the horizontal one (with two
    distinct letters, the alphabetically first steps left and the second
    steps right, so equal amounts of each sit in the middle).
    Returns {label: (x, y)} in points, or None if there are more than two
    distinct letters (a grid would put different nodes at the same spot).
    """
    letters = sorted({c for label in node_ids for c in label[1:]})
    if len(letters) > 2:
        return None
    step = {c: (i - (len(letters) - 1) / 2) * 2 for i, c in enumerate(letters)}
    return {
        label: (dx / 2 * sum(step[c] for c in label[1:]), -dy * (len(label) - 1))
        for label in node_ids
    }


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


def render(dot_source, output_path, fixed_positions=False):
    fmt = Path(output_path).suffix.lstrip(".") or "png"
    # neato -n uses the pos attributes as given; dot does its own layout
    cmd = ["neato", "-n"] if fixed_positions else ["dot"]
    try:
        subprocess.run(
            cmd + [f"-T{fmt}", "-o", str(output_path)],
            input=dot_source,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        sys.exit(f"error: Graphviz '{cmd[0]}' executable not found; please install graphviz.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: '{cmd[0]}' failed with exit code {e.returncode}")


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
        "--max-width",
        type=float,
        default=8.0,
        help="Line width (points) of the edge with the largest count; other edges "
        "are proportional to their count, with a 0.5 minimum (default: 8). Must be >= 1.",
    )
    parser.add_argument(
        "--color",
        action="append",
        metavar="LETTER=COLOR",
        help="Color for a letter, e.g. --color I=red --color P=#0080ff (repeatable). "
        "Each node is filled with the average color of the letters in its label "
        "(after the first character). Letters without a color are ignored.",
    )
    parser.add_argument(
        "--no-edge-numbers",
        action="store_true",
        help="Do not print the count next to each edge (line thickness still shows it).",
    )
    parser.add_argument(
        "--numbers",
        choices=["none", "entered", "terminated"],
        default="none",
        help="Number inside each node: 'none' (default); 'entered' = total that "
        "arrived at the node (sum of incoming edges; the root shows its own count); "
        "'terminated' = percentage of the root count that stopped at the node "
        "(entered minus what continued to children; blank if 0), so the shown "
        "percentages sum to 100%%.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Instead of drawing an image, check that each node's children's "
        "counts sum to no more than the node's own count, and report violations.",
    )
    args = parser.parse_args()
    if args.max_width < 1:
        parser.error("--max-width must be >= 1")

    nodes = parse_pathways(args.pathways)

    if args.check:
        ok = check_pathways(nodes)
        sys.exit(0 if ok else 1)

    root_id, node_ids, edges = build_graph(nodes)
    try:
        letter_colors = parse_letter_colors(args.color)
    except ValueError as e:
        parser.error(str(e))
    positions = compute_positions(node_ids)
    dot_source = to_dot(root_id, node_ids, edges, args.max_width, letter_colors,
                        args.numbers, nodes[next(iter(nodes))], positions,
                        not args.no_edge_numbers)

    if args.dot_out:
        Path(args.dot_out).write_text(dot_source)

    if not args.output:
        args.output = args.pathways + ".png"

    render(dot_source, args.output, fixed_positions=positions is not None)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
