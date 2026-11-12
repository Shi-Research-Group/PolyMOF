#!/usr/bin/env python3
"""
plot_molecule_graph.py

Read a LAMMPS molecule template file (as produced by data2molecule.py /
react_molecule.py) and draw a simple 2D graph image showing how the atoms
are bonded. Nodes are labeled with each atom's type (as given in the
"Types" section of the file). Since the "Coords" section in these template
files is not reliable (especially for post-reaction templates), a flat 2D
layout is generated instead: rings are drawn as regular polygons and
branches are attached radially outward, so that all bonds are drawn with
equal length wherever that is geometrically possible.

Usage:
    python3 plot_molecule_graph.py molecule.lammps
    python3 plot_molecule_graph.py ZN-IM-post.molecule_template -o post.png
"""

import argparse
import math
import re
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

SECTION_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z ]*$")

DEFAULT_NODE_COLOR = "#a6cee3"
INITIATOR_NODE_COLOR = "#fb9a99"
EDGE_NODE_COLOR = "#fdbf6f"
IMAGE_DPI = 150


def parse_sections(path):
    """Split a LAMMPS molecule template file into named sections.

    Returns a dict mapping section name (e.g. "Types", "Bonds") to a list
    of its data lines (comments/blank lines stripped).
    """
    sections = {}
    current = None
    with open(path, "r") as fh:
        lines = [line.split("#", 1)[0].rstrip("\n") for line in fh]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Header line/count lines (e.g. "10 atoms", "9 bonds") appear before
        # any section body; skip them - we only care about named sections.
        if SECTION_HEADER_RE.match(line) and not line[0].isdigit():
            section_name = line
            # Section body follows, possibly after a blank line, and ends at
            # the next blank line (or end of file).
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            body = []
            while i < n and lines[i].strip():
                body.append(lines[i].strip())
                i += 1
            sections[section_name] = body
        else:
            i += 1

    return sections


def parse_types(sections):
    """Return a dict mapping atom_id (int) -> type label (str)."""
    types = {}
    for line in sections.get("Types", []):
        tokens = line.split()
        atom_id = int(tokens[0])
        atom_type = tokens[1]
        types[atom_id] = atom_type
    return types


def parse_bonds(sections):
    """Return a list of (atom_id1, atom_id2) tuples from the Bonds section.

    Handles both the named-type format ("1 nc-cc 1 2") and the numeric-type
    format ("1 1 1 2") - in both cases the last two whitespace-separated
    tokens on the line are the bonded atom IDs.
    """
    bonds = []
    for line in sections.get("Bonds", []):
        tokens = line.split()
        atom_id1 = int(tokens[-2])
        atom_id2 = int(tokens[-1])
        bonds.append((atom_id1, atom_id2))
    return bonds


def build_graph(path):
    sections = parse_sections(path)

    if "Types" not in sections:
        sys.exit(f"Error: no 'Types' section found in '{path}'.")
    if "Bonds" not in sections:
        sys.exit(f"Error: no 'Bonds' section found in '{path}'.")

    types = parse_types(sections)
    bonds = parse_bonds(sections)

    graph = nx.Graph()
    for atom_id, atom_type in types.items():
        graph.add_node(atom_id, type=atom_type)
    graph.add_edges_from(bonds)

    return graph


def parse_map_file(path):
    """Parse a LAMMPS bond/react map file.

    Returns a tuple (initiator_ids, edge_ids, equivalences):
      - initiator_ids: set of atom IDs (in the pre-reaction template's
        numbering) marked as "InitiatorIDs".
      - edge_ids: set of atom IDs (pre-reaction numbering) marked as
        "EdgeIDs".
      - equivalences: dict mapping pre-reaction atom ID -> post-reaction
        atom ID, from the "Equivalences" section.
    """
    sections = parse_sections(path)

    initiator_ids = {int(line.split()[0]) for line in sections.get("InitiatorIDs", [])}
    edge_ids = {int(line.split()[0]) for line in sections.get("EdgeIDs", [])}

    equivalences = {}
    for line in sections.get("Equivalences", []):
        tokens = line.split()
        pre_id, post_id = int(tokens[0]), int(tokens[1])
        equivalences[pre_id] = post_id

    return initiator_ids, edge_ids, equivalences


def map_positions_through_equivalences(target_graph, reference_pos, equivalences):
    """Position every atom in `target_graph` using `reference_pos`, a
    position dict keyed by the *reference* template's atom IDs, via a
    dict mapping target atom ID -> reference atom ID (`equivalences`).

    Atoms in `target_graph` with no equivalence entry (e.g. atoms deleted
    or newly created during the reaction) are attached at unit distance
    from an already-positioned neighbor, radiating outward from the
    centroid of the mapped atoms - the same approach used for ordinary
    branch atoms elsewhere in this script.
    """
    pos = {}
    visited = set()
    for node in target_graph.nodes:
        ref_id = equivalences.get(node)
        if ref_id is not None and ref_id in reference_pos:
            pos[node] = reference_pos[ref_id]
            visited.add(node)

    if visited:
        centroid = (
            sum(p[0] for p in pos.values()) / len(pos),
            sum(p[1] for p in pos.values()) / len(pos),
        )
    else:
        centroid = (0.0, 0.0)

    directions = {}
    for node in visited:
        vx, vy = pos[node][0] - centroid[0], pos[node][1] - centroid[1]
        norm = math.hypot(vx, vy)
        directions[node] = (vx / norm, vy / norm) if norm > 1e-9 else (1.0, 0.0)

    _place_branches(target_graph, pos, directions, visited)

    # Anything still unplaced (e.g. a whole disconnected fragment with no
    # equivalence at all) gets its own independent layout, offset aside.
    remaining = set(target_graph.nodes) - set(pos)
    if remaining:
        sub_pos = compute_equal_length_layout(target_graph.subgraph(remaining))
        offset_x = max((p[0] for p in pos.values()), default=0.0) + 2.0
        for node, (x, y) in sub_pos.items():
            pos[node] = (x + offset_x, y)

    return pos


def _regular_polygon_radius(n):
    """Circumradius of a regular n-gon with unit-length sides."""
    return 0.5 / math.sin(math.pi / n) if n > 2 else 0.5


def _compute_ring_layout(ring, entry_atom, anchor_pos, outward_dir):
    """Compute positions/outward-directions for `ring` (a list of atom IDs
    in cyclic order) laid out as a regular polygon with unit-length sides,
    attached via a unit-length bond from `anchor_pos` to `entry_atom`,
    continuing outward along `outward_dir`. Returns (positions, directions)
    dicts, without mutating any shared state.
    """
    n = len(ring)
    radius = _regular_polygon_radius(n)
    theta = math.atan2(outward_dir[1], outward_dir[0])
    entry_angle = theta + math.pi
    entry_index = ring.index(entry_atom)

    center_x = anchor_pos[0] + outward_dir[0] * (1.0 + radius)
    center_y = anchor_pos[1] + outward_dir[1] * (1.0 + radius)

    step = 2 * math.pi / n
    positions = {}
    out_directions = {}
    for offset, node in enumerate(ring):
        angle = entry_angle + step * (offset - entry_index)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions[node] = (x, y)
        out_directions[node] = (math.cos(angle), math.sin(angle))
    return positions, out_directions


def _compute_fused_ring_layout(ring, atom_a, atom_b, pos, avoid_point):
    """Compute positions/outward-directions for the not-yet-placed members
    of `ring` (a list of atom IDs in cyclic order), given that `atom_a`
    and `atom_b` are two ring-adjacent atoms that are already placed (a
    shared, unit-length edge with some other, already-drawn ring). The
    rest of `ring` is laid out as a regular, unit-bond-length polygon on
    the far side of that shared edge from `avoid_point`, so ortho-fused
    ring systems (e.g. a 5-membered ring fused to a 6-membered ring, as in
    imidazole-fused six-membered rings) keep every bond unit length
    instead of collapsing the second ring into overlapping branch atoms.

    Returns (positions, directions) dicts covering every atom in `ring`
    (including `atom_a`/`atom_b`, unchanged), without mutating any shared
    state.
    """
    n = len(ring)
    idx_a = ring.index(atom_a)
    idx_b = ring.index(atom_b)

    if ring[(idx_a + 1) % n] == atom_b:
        ordered = ring[idx_a:] + ring[:idx_a]
    elif ring[(idx_b + 1) % n] == atom_a:
        rev = list(reversed(ring))
        idx_a2 = rev.index(atom_a)
        ordered = rev[idx_a2:] + rev[:idx_a2]
    else:
        raise ValueError("atom_a/atom_b are not adjacent in ring")

    ax, ay = pos[atom_a]
    bx, by = pos[atom_b]
    edge_dx, edge_dy = bx - ax, by - ay
    mid_x, mid_y = (ax + bx) / 2.0, (ay + by) / 2.0
    perp_x, perp_y = -edge_dy, edge_dx  # unit, since edge is unit length

    radius = _regular_polygon_radius(n)
    apothem = radius * math.cos(math.pi / n)

    candidate_centers = [
        (mid_x + perp_x * apothem, mid_y + perp_y * apothem),
        (mid_x - perp_x * apothem, mid_y - perp_y * apothem),
    ]
    center_x, center_y = max(
        candidate_centers,
        key=lambda c: math.hypot(c[0] - avoid_point[0], c[1] - avoid_point[1]),
    )

    angle_a = math.atan2(ay - center_y, ax - center_x)
    angle_b = math.atan2(by - center_y, bx - center_x)
    step = angle_b - angle_a
    # Normalize to the +/- 2*pi/n step matching this polygon's winding.
    while step <= -math.pi:
        step += 2 * math.pi
    while step > math.pi:
        step -= 2 * math.pi
    step = math.copysign(2 * math.pi / n, step)

    positions = {}
    out_directions = {}
    for offset, node in enumerate(ordered):
        angle = angle_a + step * offset
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        positions[node] = (x, y)
        out_directions[node] = (math.cos(angle), math.sin(angle))
    return positions, out_directions


def _find_collision_free_angle(base_angle, place_fn, occupied_positions, min_dist=0.5):
    """Try `place_fn(angle)` (which returns a candidate {atom: (x, y)} dict
    for one or more new atoms) at `base_angle` and, if any candidate
    position collides with an already-occupied position, at a sequence of
    nearby angles instead. Returns (angle, candidate_positions) for the
    first collision-free attempt found, or the original if none is.
    """
    offsets_deg = [0] + [sign * step for step in range(10, 180, 10) for sign in (1, -1)]
    fallback = None
    for offset_deg in offsets_deg:
        angle = base_angle + math.radians(offset_deg)
        candidates = place_fn(angle)
        collision = any(
            math.hypot(cx - ox, cy - oy) < min_dist
            for cx, cy in candidates.values()
            for ox, oy in occupied_positions
        )
        if fallback is None:
            fallback = (angle, candidates)
        if not collision:
            return angle, candidates
    return fallback


def _place_branches(graph, pos, directions, visited):
    """Breadth-first attach remaining atoms at unit distance from their
    already-placed parent, radiating outward. Each node's outgoing
    direction is inherited from the direction it was reached from (the
    parent's own outward direction), and children of the same parent are
    fanned out over a small arc so they don't overlap.

    If a not-yet-placed neighbor belongs to a ring (as found by cycle
    detection on the whole graph) that hasn't been placed yet, the entire
    ring is laid out as a regular, unit-bond-length polygon attached at
    that neighbor, rather than treating just that one atom as a simple
    branch step - this keeps every ring in the molecule (not just the
    largest one) drawn as a proper regular polygon, and keeps rings that
    are only connected by a bridge (rather than fused/sharing atoms) from
    being collapsed on top of each other.

    Whenever a candidate placement would land on top of (or very close to)
    an already-placed, unrelated atom - which can happen coincidentally in
    larger/branchier molecules - the placement angle is nudged until the
    collision is resolved.
    """
    rings_by_atom = {}
    for ring in nx.cycle_basis(graph):
        for atom in ring:
            rings_by_atom.setdefault(atom, ring)

    placed_rings = set()

    queue = deque(visited)
    while queue:
        node = queue.popleft()
        base_dx, base_dy = directions[node]
        base_angle = math.atan2(base_dy, base_dx)

        children = [nbr for nbr in graph.neighbors(node) if nbr not in visited]
        k = len(children)
        if k == 0:
            continue

        spread = math.radians(70)
        for idx, child in enumerate(children):
            if child in visited:
                continue  # may have been placed by an earlier sibling's ring attachment
            if k == 1:
                target_angle = base_angle
            else:
                target_angle = base_angle - spread / 2 + spread * idx / (k - 1)

            child_ring = rings_by_atom.get(child)
            shared_atoms = (
                [a for a in child_ring if a in visited]
                if child_ring is not None
                else []
            )
            ring_already_placed = child_ring is not None and (
                id(child_ring) in placed_rings or bool(shared_atoms)
            )

            # An ortho-fused ring (sharing exactly one edge - i.e. exactly
            # two ring-adjacent atoms - with an already-placed ring, e.g. a
            # 5-membered ring fused to a 6-membered ring) can still be
            # drawn with every bond at unit length: lay the rest of the
            # ring out as a regular polygon on the far side of that shared
            # edge, instead of collapsing it into plain branch atoms.
            fused_edge = None
            if child_ring is not None and len(shared_atoms) == 2:
                a0, a1 = shared_atoms
                n_ring = len(child_ring)
                i0 = child_ring.index(a0)
                if child_ring[(i0 + 1) % n_ring] == a1 or child_ring[(i0 - 1) % n_ring] == a1:
                    fused_edge = (a0, a1)

            occupied = list(pos.values())
            if fused_edge is not None and id(child_ring) not in placed_rings:
                avoid_x = sum(p[0] for p in pos.values()) / len(pos)
                avoid_y = sum(p[1] for p in pos.values()) / len(pos)
                final_positions, final_directions = _compute_fused_ring_layout(
                    child_ring, fused_edge[0], fused_edge[1], pos, (avoid_x, avoid_y)
                )
                pos.update(final_positions)
                directions.update(final_directions)
                visited.update(child_ring)
                placed_rings.add(id(child_ring))
                for atom in child_ring:
                    queue.append(atom)
            elif child_ring is not None and not ring_already_placed:
                place_fn = lambda angle, node=node, child=child, child_ring=child_ring: (
                    _compute_ring_layout(
                        child_ring, child, pos[node], (math.cos(angle), math.sin(angle))
                    )[0]
                )
                _, ring_positions = _find_collision_free_angle(target_angle, place_fn, occupied)
                chosen_dx = ring_positions[child][0] - pos[node][0]
                chosen_dy = ring_positions[child][1] - pos[node][1]
                norm = math.hypot(chosen_dx, chosen_dy)
                outward = (chosen_dx / norm, chosen_dy / norm)
                final_positions, final_directions = _compute_ring_layout(
                    child_ring, child, pos[node], outward
                )
                pos.update(final_positions)
                directions.update(final_directions)
                visited.update(child_ring)
                placed_rings.add(id(child_ring))
                for atom in child_ring:
                    queue.append(atom)
            else:
                place_fn = lambda angle, node=node, child=child: {
                    child: (
                        pos[node][0] + math.cos(angle),
                        pos[node][1] + math.sin(angle),
                    )
                }
                _, candidates = _find_collision_free_angle(target_angle, place_fn, occupied)
                child_pos = candidates[child]
                pos[child] = child_pos
                directions[child] = (
                    child_pos[0] - pos[node][0],
                    child_pos[1] - pos[node][1],
                )
                visited.add(child)
                queue.append(child)


def _layout_component(component_graph):
    """Compute a unit-bond-length 2D layout for a single connected
    component: the largest ring (if any) is drawn as a regular polygon with
    unit-length sides, and every other atom is attached at unit distance
    from its neighbor, radiating outward from the ring (or from an
    arbitrary root, if the component has no ring).
    """
    pos = {}
    directions = {}
    visited = set()

    cycles = nx.cycle_basis(component_graph)
    if cycles:
        main_ring = max(cycles, key=len)
        n = len(main_ring)
        # Regular n-gon circumradius that gives unit-length edges.
        radius = _regular_polygon_radius(n)
        for i, node in enumerate(main_ring):
            angle = 2 * math.pi * i / n
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
            directions[node] = (math.cos(angle), math.sin(angle))
            visited.add(node)
    else:
        root = next(iter(component_graph.nodes))
        pos[root] = (0.0, 0.0)
        directions[root] = (0.0, 1.0)
        visited.add(root)

    _place_branches(component_graph, pos, directions, visited)

    # Any nodes not reached yet belong to extra, fused/independent cycles
    # within this component; place them as their own mini polygon (best
    # effort - equal bond lengths are not always geometrically possible
    # once rings share atoms or edges).
    remaining = set(component_graph.nodes) - visited
    while remaining:
        sub_cycles = nx.cycle_basis(component_graph.subgraph(remaining))
        seed = max(sub_cycles, key=len) if sub_cycles else [next(iter(remaining))]
        offset_x = max((p[0] for p in pos.values()), default=0.0) + 2.0
        n = len(seed)
        radius = _regular_polygon_radius(n)
        for i, node in enumerate(seed):
            angle = 2 * math.pi * i / n
            pos[node] = (offset_x + radius * math.cos(angle), radius * math.sin(angle))
            directions[node] = (math.cos(angle), math.sin(angle))
            visited.add(node)
            remaining.discard(node)
        _place_branches(component_graph, pos, directions, visited)
        remaining = set(component_graph.nodes) - visited

    return pos


def compute_equal_length_layout(graph):
    """Compute a 2D layout for the whole (possibly disconnected) graph in
    which every bond has unit length, wherever that is geometrically
    possible (always true for trees and simple, unfused rings; only
    approximate for fused/bridged ring systems).
    """
    pos = {}
    x_offset = 0.0
    for component_nodes in nx.connected_components(graph):
        component_graph = graph.subgraph(component_nodes)
        component_pos = _layout_component(component_graph)

        min_x = min(p[0] for p in component_pos.values())
        max_x = max(p[0] for p in component_pos.values())
        shift = x_offset - min_x
        for node, (x, y) in component_pos.items():
            pos[node] = (x + shift, y)
        x_offset += (max_x - min_x) + 2.0  # gap between components

    return pos


def _scale_to_fit(fig_w, fig_h, max_width_px, max_height_px, dpi=IMAGE_DPI):
    """Scale down (fig_w, fig_h) (inches) as needed so that the resulting
    image, when saved at `dpi`, does not exceed `max_width_px` and/or
    `max_height_px` (in pixels), preserving the original aspect ratio.
    Either limit may be None to leave that dimension unconstrained.
    """
    scale = 1.0
    if max_width_px is not None:
        width_px = fig_w * dpi
        if width_px > max_width_px:
            scale = min(scale, max_width_px / width_px)
    if max_height_px is not None:
        height_px = fig_h * dpi
        if height_px > max_height_px:
            scale = min(scale, max_height_px / height_px)
    return fig_w * scale, fig_h * scale


def plot_graph(
    graph,
    title,
    output_path,
    pos=None,
    node_categories=None,
    label_by="type",
    max_width=None,
    max_height=None,
):
    """Draw `graph` to `output_path`.

    pos: optional pre-computed {atom_id: (x, y)} layout; if omitted, a fresh
    equal-bond-length layout is computed for `graph`.
    node_categories: optional {atom_id: "initiator"|"edge"} dict used to
    color-code initiator/edge atoms differently from ordinary atoms; a
    legend is added whenever this is given.
    label_by: "type" (default) to label nodes with their atom type, or
    "id" to label nodes with their atom ID/number instead.
    max_width, max_height: optional maximum output image size, in pixels.
    If the image would otherwise exceed either limit, it is scaled down
    (preserving its aspect ratio) so that it fits within both.
    """
    if pos is None:
        pos = compute_equal_length_layout(graph)
    if label_by == "id":
        labels = {n: str(n) for n in graph.nodes}
    else:
        labels = {n: graph.nodes[n]["type"] for n in graph.nodes}

    # Size the figure to match the data's true aspect ratio (plus a little
    # padding/margin for node markers) so that unit-length bonds are never
    # stretched or squeezed into a fixed square canvas.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width = max(xs) - min(xs) if len(xs) > 1 else 0.0
    height = max(ys) - min(ys) if len(ys) > 1 else 0.0
    margin = 1.5  # room for node markers/labels at the edges, in data units
    width += margin
    height += margin

    scale = 1.3  # inches per data unit
    fig_w = max(width * scale, 4.0)
    fig_h = max(height * scale, 4.0)
    fig_w, fig_h = _scale_to_fit(fig_w, fig_h, max_width, max_height)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.5, edge_color="gray")

    node_categories = node_categories or {}
    node_list = list(graph.nodes)
    color_by_category = {
        "initiator": INITIATOR_NODE_COLOR,
        "edge": EDGE_NODE_COLOR,
        None: DEFAULT_NODE_COLOR,
    }
    node_colors = [
        color_by_category[node_categories.get(n)] for n in node_list
    ]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=node_list,
        node_size=900,
        node_color=node_colors,
        edgecolors="black",
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=10)

    if node_categories:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=12,
                label=label,
            )
            for label, color in (
                ("initiator atom", INITIATOR_NODE_COLOR),
                ("edge atom", EDGE_NODE_COLOR),
                ("other atom", DEFAULT_NODE_COLOR),
            )
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    # Enforce a 1:1 data aspect ratio so equal-length bonds are rendered
    # with visually equal length, regardless of the figure's width/height.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=IMAGE_DPI)
    plt.close(fig)


def run_pre_post(
    pre_path,
    post_path,
    map_path,
    pre_output,
    post_output,
    label_by="type",
    max_width=None,
    max_height=None,
):
    """Build both the pre- and post-reaction bonding graph images, placing
    each pre-template atom at the same position as its equivalent
    post-template atom (per the map file), and color-coding initiator and
    edge atoms in both images.
    """
    initiator_ids, edge_ids, equivalences = parse_map_file(map_path)

    pre_graph = build_graph(pre_path)
    post_graph = build_graph(post_path)

    # Layout is computed from the post-reaction graph (topologically the
    # more complete structure, e.g. it includes the newly formed bond), then
    # the pre-reaction atoms are placed at their equivalent post positions.
    pos_post = compute_equal_length_layout(post_graph)
    pos_pre = map_positions_through_equivalences(pre_graph, pos_post, equivalences)

    # Initiator/edge IDs in the map file use pre-template numbering; map
    # them onward to post-template IDs via the same equivalences.
    post_initiator_ids = {equivalences[i] for i in initiator_ids if i in equivalences}
    post_edge_ids = {equivalences[i] for i in edge_ids if i in equivalences}

    def categorize(atom_id, initiators, edges):
        if atom_id in initiators:
            return "initiator"
        if atom_id in edges:
            return "edge"
        return None

    pre_categories = {n: categorize(n, initiator_ids, edge_ids) for n in pre_graph.nodes}
    post_categories = {
        n: categorize(n, post_initiator_ids, post_edge_ids) for n in post_graph.nodes
    }

    plot_graph(
        pre_graph,
        title=Path(pre_path).name,
        output_path=pre_output,
        pos=pos_pre,
        node_categories=pre_categories,
        label_by=label_by,
        max_width=max_width,
        max_height=max_height,
    )
    plot_graph(
        post_graph,
        title=Path(post_path).name,
        output_path=post_output,
        pos=pos_post,
        node_categories=post_categories,
        label_by=label_by,
        max_width=max_width,
        max_height=max_height,
    )

    print(
        f"Wrote pre-reaction graph ({pre_graph.number_of_nodes()} atoms / "
        f"{pre_graph.number_of_edges()} bonds) to '{pre_output}'."
    )
    print(
        f"Wrote post-reaction graph ({post_graph.number_of_nodes()} atoms / "
        f"{post_graph.number_of_edges()} bonds) to '{post_output}'."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read a LAMMPS molecule template file and draw a 2D bonding "
            "graph image, with nodes labeled by atom type."
        )
    )
    parser.add_argument(
        "template_file",
        nargs="?",
        default=None,
        help="Path to a single LAMMPS molecule template file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output image path (e.g. molecule.png). Defaults to the input "
            "filename with a .png extension."
        ),
    )
    parser.add_argument(
        "--pre",
        default=None,
        help="Pre-reaction molecule template file (used together with --post/--map).",
    )
    parser.add_argument(
        "--post",
        default=None,
        help="Post-reaction molecule template file (used together with --pre/--map).",
    )
    parser.add_argument(
        "--map",
        dest="map_file",
        default=None,
        help="LAMMPS bond/react map file (used together with --pre/--post).",
    )
    parser.add_argument(
        "--pre-output",
        default=None,
        help="Output image path for the pre-reaction graph. Defaults to <pre>.png.",
    )
    parser.add_argument(
        "--post-output",
        default=None,
        help="Output image path for the post-reaction graph. Defaults to <post>.png.",
    )
    parser.add_argument(
        "-n",
        "--numbers",
        action="store_true",
        help=(
            "Label graph nodes with their atom ID/number instead of their "
            "atom type."
        ),
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        metavar="PIXELS",
        help=(
            "Maximum output image width, in pixels. If the image would "
            "otherwise be wider, it is scaled down (preserving its aspect "
            "ratio) to fit."
        ),
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=None,
        metavar="PIXELS",
        help=(
            "Maximum output image height, in pixels. If the image would "
            "otherwise be taller, it is scaled down (preserving its aspect "
            "ratio) to fit."
        ),
    )
    args = parser.parse_args()

    label_by = "id" if args.numbers else "type"

    for label, value in (("--max-width", args.max_width), ("--max-height", args.max_height)):
        if value is not None and value <= 0:
            sys.exit(f"Error: {label} must be a positive number of pixels.")

    pre_post_args = (args.pre, args.post, args.map_file)
    using_pre_post = any(v is not None for v in pre_post_args)

    if using_pre_post and args.template_file is not None:
        sys.exit(
            "Error: pass either a single template_file, or --pre/--post/--map "
            "together - not both."
        )

    if using_pre_post:
        if any(v is None for v in pre_post_args):
            sys.exit("Error: --pre, --post, and --map must all be given together.")

        for label, p in (("--pre", args.pre), ("--post", args.post), ("--map", args.map_file)):
            if not Path(p).is_file():
                sys.exit(f"Error: file not found for {label}: {p}")

        pre_output = Path(args.pre_output) if args.pre_output else Path(args.pre).with_suffix(".png")
        post_output = (
            Path(args.post_output) if args.post_output else Path(args.post).with_suffix(".png")
        )
        run_pre_post(
            args.pre,
            args.post,
            args.map_file,
            pre_output,
            post_output,
            label_by=label_by,
            max_width=args.max_width,
            max_height=args.max_height,
        )
        return

    if args.template_file is None:
        sys.exit(
            "Error: provide a template_file, or use --pre/--post/--map together."
        )

    template_path = Path(args.template_file)
    if not template_path.is_file():
        sys.exit(f"Error: file not found: {template_path}")

    output_path = Path(args.output) if args.output else template_path.with_suffix(".png")

    graph = build_graph(template_path)
    plot_graph(
        graph,
        title=template_path.name,
        output_path=output_path,
        label_by=label_by,
        max_width=args.max_width,
        max_height=args.max_height,
    )

    print(
        f"Wrote bonding graph for {graph.number_of_nodes()} atoms / "
        f"{graph.number_of_edges()} bonds to '{output_path}'."
    )



if __name__ == "__main__":
    main()
