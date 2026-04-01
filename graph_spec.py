# graph_spec.py
# ── Shared graph specification used by both source.py and the learning model ──

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MotifSpec:
    """Specification for a single motif subgraph."""
    n:          int                    # number of neurons
    adjacency:  np.ndarray             # (n,n) bool — A[i,j]=True means j->i
    label:      str        = ""
    eps_init:   float      = 0.10      # initial epsilon (for numpy sim)
    delta_init: float      = 0.50      # initial delta   (for numpy sim)

    def __post_init__(self):
        if not self.label:
            self.label = f"Motif({self.n})"

    @classmethod
    def cyclic(cls, n: int, label: str = "", **kwargs) -> "MotifSpec":
        """Create a cyclic tournament on n nodes."""
        A = np.zeros((n, n), dtype=bool)
        for j in range(n):
            A[(j + 1) % n, j] = True
        return cls(n=n, adjacency=A, label=label or f"Cycle({n})", **kwargs)

    @classmethod
    def from_edge_list(cls, n: int, edges: List[tuple],
                       label: str = "", **kwargs) -> "MotifSpec":
        """
        Create from explicit edge list.
        edges: list of (source, target) tuples — source->target
        Stored as A[target, source] = True to match CTLN convention.
        """
        A = np.zeros((n, n), dtype=bool)
        for src, tgt in edges:
            A[tgt, src] = True
        return cls(n=n, adjacency=A, label=label or f"Custom({n})", **kwargs)


@dataclass
class NetworkSpec:
    """
    Full network specification: a list of motifs plus inter-motif coupling.
    This is the single source of truth consumed by both the numpy simulator
    and the PyTorch learning model.
    """
    motifs:          List[MotifSpec]
    delta_cross:     float             = 0.05
    # Per-pair cross coupling — overrides delta_cross if provided
    # cross_deltas[i][j] = coupling from motif j to motif i
    cross_deltas:    Optional[np.ndarray] = None

    def __post_init__(self):
        k = len(self.motifs)
        if self.cross_deltas is None:
            self.cross_deltas = np.full((k, k), self.delta_cross)
        # Ensure diagonal is zero (no self-cross coupling)
        np.fill_diagonal(self.cross_deltas, 0.0)

    @property
    def n_total(self) -> int:
        return sum(m.n for m in self.motifs)

    @property
    def k(self) -> int:
        return len(self.motifs)

    @property
    def slices(self) -> List[slice]:
        starts = np.concatenate(
            ([0], np.cumsum([m.n for m in self.motifs])[:-1])).astype(int)
        return [slice(s, s + m.n)
                for s, m in zip(starts, self.motifs)]

    @property
    def labels(self) -> List[str]:
        return [m.label for m in self.motifs]

    def to_numpy_W(self) -> np.ndarray:
        """
        Build the full numpy weight matrix using each motif's
        eps_init and delta_init. Used by source.py simulator.
        """
        N      = self.n_total
        W      = np.zeros((N, N))
        slices = self.slices

        for i, (si, mi) in enumerate(zip(slices, self.motifs)):
            for j, (sj, mj) in enumerate(zip(slices, self.motifs)):
                if i == j:
                    # Within-motif block
                    A = mi.adjacency.astype(float)
                    block = (np.where(A,
                                      -1.0 + mi.eps_init,
                                      -1.0 - mi.delta_init)
                             * (1 - np.eye(mi.n)))
                    W[si, si] = block
                else:
                    # Cross-motif block — all non-edges
                    dc = self.cross_deltas[i, j]
                    W[si, sj] = np.full((mi.n, mj.n), -1.0 - dc)
        return W

    def to_torch_adjacency(self) -> torch.Tensor:
        """
        Build the full (N,N) boolean adjacency tensor for the PyTorch model.
        Cross-motif connections are all False (non-edges).
        """
        N = self.n_total
        A = torch.zeros(N, N, dtype=torch.bool)
        for sl, m in zip(self.slices, self.motifs):
            A[sl, sl] = torch.from_numpy(m.adjacency)
        return A

    def summary(self):
        print(f"NetworkSpec: {self.k} motifs, {self.n_total} neurons total")
        for i, m in enumerate(self.motifs):
            n_edges = m.adjacency.sum()
            print(f"  [{i}] {m.label}: {m.n} neurons, "
                  f"{n_edges} edges, "
                  f"eps={m.eps_init}, delta={m.delta_init}")
        print(f"  delta_cross matrix:\n{np.round(self.cross_deltas, 3)}")


# Add to graph_spec.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_network_graph(spec,
                       W=None,
                       theta=None,
                       title="",
                       show_weights=True,
                       show_theta=True,
                       node_scale=1.0,
                       figsize=(10, 8)):
    """
    Visualize the network as a directed graph.

    Parameters
    ----------
    spec         : NetworkSpec
    W            : (N,N) numpy array of weights — if None uses spec defaults
    theta        : (N,) numpy array of per-neuron drive — if None uses 0.1
    show_weights : annotate edges with weight values
    show_theta   : show theta value inside each node
    node_scale   : scale node sizes (1.0 = default)
    """
    N       = spec.n_total
    slices  = spec.slices
    labels  = spec.labels
    k       = spec.k

    # ── Build weight matrix if not provided ──────────────────────────────
    if W is None:
        W = spec.to_numpy_W()
    if theta is None:
        theta = np.full(N, 0.1)

    # ── Build directed graph ──────────────────────────────────────────────
    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(N):
            if i != j:
                G.add_edge(j, i, weight=W[i, j])
                # Note: W[i,j] = weight FROM j TO i
                # In graph convention: edge j->i

    # ── Node layout ───────────────────────────────────────────────────────
    # Each motif gets its own circular cluster, clusters spread in a ring
    pos = {}

    if k == 1:
        # Single motif — just place in a circle
        m   = spec.motifs[0]
        sl  = slices[0]
        for local_idx, global_idx in enumerate(range(sl.start, sl.stop)):
            angle = 2 * np.pi * local_idx / m.n
            pos[global_idx] = (np.cos(angle), np.sin(angle))
    else:
        # Multiple motifs — place each cluster at a position on a big ring
        cluster_radius = 3.5
        motif_radius   = 0.8

        for motif_idx, (sl, m) in enumerate(zip(slices, spec.motifs)):
            # Center of this motif cluster
            cluster_angle = 2 * np.pi * motif_idx / k
            cx = cluster_radius * np.cos(cluster_angle)
            cy = cluster_radius * np.sin(cluster_angle)

            # Neurons within the motif arranged in a small circle
            for local_idx, global_idx in enumerate(
                    range(sl.start, sl.stop)):
                neuron_angle = (2 * np.pi * local_idx / m.n
                                - np.pi / 2)  # start from top
                px = cx + motif_radius * np.cos(neuron_angle)
                py = cy + motif_radius * np.sin(neuron_angle)
                pos[global_idx] = (px, py)

    # ── Color scheme ──────────────────────────────────────────────────────
    # Nodes: colored by motif membership
    from itertools import cycle
    motif_palette = ["#c0392b", "#1a5276", "#1e8449",
                     "#7d3c98", "#d35400", "#117a65",
                     "#884ea0", "#1a5276", "#148f77"]

    node_colors = []
    node_sizes  = []
    for ni in range(N):
        for motif_idx, sl in enumerate(slices):
            if sl.start <= ni < sl.stop:
                node_colors.append(motif_palette[motif_idx
                                                  % len(motif_palette)])
                # Size proportional to theta
                base_size = 800 * node_scale
                node_sizes.append(base_size * (1 + 2 * theta[ni]))
                break

    # Edges: colored and weighted by value
    # Edge weights are all negative — more negative = stronger inhibition
    # Color: red = near 0 (edge, weak inhibition)
    #        blue = very negative (non-edge, strong inhibition)
    all_weights = np.array([W[i, j] for j, i in G.edges()
                             if i != j])
    w_min = all_weights.min()
    w_max = all_weights.max()
    norm  = Normalize(vmin=w_min, vmax=w_max)
    cmap  = plt.cm.RdBu  # red=less negative, blue=more negative

    edge_colors = []
    edge_widths = []
    edge_styles = []
    for (src, tgt) in G.edges():
        w = W[tgt, src]  # weight from src to tgt
        edge_colors.append(cmap(norm(w)))

        # Thicker lines for edges (less inhibitory = graph edges)
        is_graph_edge = spec.to_torch_adjacency()[tgt, src].item()
        edge_widths.append(2.5 if is_graph_edge else 0.8)
        edge_styles.append("solid" if is_graph_edge else "dashed")

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw edges — draw non-edges first (behind), then graph edges
    A_full = spec.to_torch_adjacency().numpy()

    # Non-edges (dashed, thin, low alpha)
    non_edge_list = [(src, tgt) for src, tgt in G.edges()
                     if not A_full[tgt, src]]
    if non_edge_list:
        nx.draw_networkx_edges(
            G, pos,
            edgelist   = non_edge_list,
            edge_color = [cmap(norm(W[tgt, src]))
                          for src, tgt in non_edge_list],
            width      = 0.6,
            alpha      = 0.25,
            style      = "dashed",
            arrows     = True,
            arrowsize  = 8,
            arrowstyle = "-|>",
            connectionstyle = "arc3,rad=0.1",
            ax         = ax
        )

    # Graph edges (solid, thick, full opacity)
    graph_edge_list = [(src, tgt) for src, tgt in G.edges()
                       if A_full[tgt, src]]
    if graph_edge_list:
        nx.draw_networkx_edges(
            G, pos,
            edgelist   = graph_edge_list,
            edge_color = [cmap(norm(W[tgt, src]))
                          for src, tgt in graph_edge_list],
            width      = 2.5,
            alpha      = 0.9,
            style      = "solid",
            arrows     = True,
            arrowsize  = 15,
            arrowstyle = "-|>",
            connectionstyle = "arc3,rad=0.15",
            ax         = ax
        )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color = node_colors,
        node_size  = node_sizes,
        alpha      = 0.9,
        ax         = ax
    )

    # ── Node labels ───────────────────────────────────────────────────────
    if show_theta:
        # Show neuron index + theta value inside node
        node_labels = {ni: f"n{ni}\nθ={theta[ni]:.2f}"
                       for ni in range(N)}
    else:
        node_labels = {ni: f"n{ni}" for ni in range(N)}

    nx.draw_networkx_labels(
        G, pos,
        labels    = node_labels,
        font_size = 7 if show_theta else 9,
        font_color = "white",
        ax        = ax
    )

    # ── Edge weight labels ────────────────────────────────────────────────
    if show_weights:
        # Only label graph edges — non-edges would clutter the plot
        edge_labels = {}
        for src, tgt in graph_edge_list:
            w = W[tgt, src]
            edge_labels[(src, tgt)] = f"{w:.2f}"

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels       = edge_labels,
            font_size         = 7,
            font_color        = "#2c3e50",
            bbox              = dict(boxstyle="round,pad=0.2",
                                     fc="white", alpha=0.7,
                                     ec="none"),
            label_pos         = 0.35,
            ax                = ax
        )

    # ── Motif boundary circles ────────────────────────────────────────────
    if k > 1:
        cluster_radius = 3.5
        for motif_idx, (sl, m) in enumerate(zip(slices, spec.motifs)):
            cluster_angle = 2 * np.pi * motif_idx / k
            cx = cluster_radius * np.cos(cluster_angle)
            cy = cluster_radius * np.sin(cluster_angle)
            circle = plt.Circle(
                (cx, cy), 1.3,
                fill      = False,
                linestyle = "--",
                linewidth = 1.2,
                color     = motif_palette[motif_idx % len(motif_palette)],
                alpha     = 0.4
            )
            ax.add_patch(circle)
            # Motif label above the cluster
            ax.text(cx, cy + 1.55, m.label,
                    ha="center", va="bottom",
                    fontsize=10, fontweight="500",
                    color=motif_palette[motif_idx % len(motif_palette)])

    # ── Colorbar for edge weights ─────────────────────────────────────────
    sm  = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02,
                        shrink=0.6)
    cbar.set_label("Weight W[i,j]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_yticks([w_min, (w_min+w_max)/2, w_max])
    cbar.ax.set_yticklabels([f"{w_min:.2f}",
                              f"{(w_min+w_max)/2:.2f}",
                              f"{w_max:.2f}"])

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = []
    for motif_idx, m in enumerate(spec.motifs):
        legend_elements.append(
            mpatches.Patch(
                color = motif_palette[motif_idx % len(motif_palette)],
                label = m.label,
                alpha = 0.9
            )
        )
    legend_elements += [
        mpatches.Patch(color="gray",  label="─── graph edge",
                       linewidth=2.5),
        mpatches.Patch(color="gray",  label="- - non-edge",
                       linewidth=0.6, linestyle="--"),
    ]
    ax.legend(handles    = legend_elements,
              loc        = "lower right",
              fontsize   = 8,
              framealpha = 0.6)

    ax.set_title(title or f"Network graph — {N} neurons, {k} motifs",
                 fontsize=12, pad=14)
    plt.tight_layout()
    plt.savefig(title + ".png")
    return fig, ax