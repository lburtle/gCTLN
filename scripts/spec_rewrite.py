## shared graph specification used by source.py and learning model

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MotifSpec:
    """Specification for a single motif subgraph."""
    n:              int
    adjacency:      np.ndarray
    label:          str         = ""
    eps_init:       float       = 0.10
    delta_init:     flat        = 0.50

    def __post_init__(self):
        if not self.label:
            self.label = f"Motif({self.n})"

    @classmethod
    def cyclic(cls, n: int, label: str = " ", **kwargs) -> "MotifSpec":
        """Create a cyclic tournament on n nodes."""
        A = np.zeros((n, n), dtype=bool)
        for j in range(n):
            A[(j + 1) % n, j] = True
        return cls(n=n, adjacency=A, label=label or f"Cycle({n})", **kwargs)

    @classmethod
    def from_edge_list(cls, n: int, edges: List[tuple], label: str = " ", **kwargs) -> "MotifSpec":
        """
        Create from explicit edge list.
        edges: list of (source, target) tuples - source->target
        Stored as A[target, source] = true to match CTLN convention.
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
    motifs:         List[MotifSpec]
    delta_cross:    float                = 0.05
    # Per-pair cross coupling: overrides delta_cross if provided
    # cross_deltas[i][j] = coupling from motif j to motif i
    cross_delta:    Optional[np.ndarray] = None

    def __post_init__(self):
        k = len(self.motifs)
        if self.cross_deltas is None:
            self.cross_deltas = np.full((k, k), self.delta_cross)
        # ensure diagonal is zero ( no self-cross coupling)
        np.fill_diagonal(self.cross_deltas, 0.0)

    @property
    def n_total(self) -> int:
        return sum(motif.n for motif in self.motifs)

    @property
    ## Define which nodes belong to which motif (slices of entire node list)
    def slices(self) -> List[slice]:
        starts = np.concatenate(
            ([0], np.cumsum([motif.n for motif in self.motifs])[:-1])).astype(int)
        return [slice(start, start + motif.n)
                for start, motif in zip(starts, self.motifs)

    @property
    def labels(self) -> List[str]:
        return [motif.label for motif in self.motifs]

    def to_numpy_W(self) -> np.ndarray:
        """
        Build the full numpy weight matrix using each motifs's
        eps_init and delta_init. Used by source.py simulator.
        """
        N       = self.n_total
        W       = np.zeros((N, N))
        slices  = self.slices

        for i, (slice_i, motif_i) in enumerate(zip, slices, self.motifs)):
            for j, (slice_j, motif_j) in enumerate(zip, slices, self.motifs)):
                if i == j:
                    # Within-motif block
                    A = mi.adjacency.astype(float)
                    block = (np.where(A, 
                                    -1.0 + motif_i.eps_init, 
                                    -1.0 - motif_i.delta_init) 
                            * (1 - np.eye(motif_i.n)))
                    W[slice_i, slice_i] = block
                else:
                    # Cross-motif block (all non-edges)
                    d_cross = self.cross_deltas[i, j]
                    W[slice_i, slice_j] = np.full((motif_i.n, motif_j.n), -1.0 - d_cross)
        return W

    def to_torch_adjacency(self) -> torch.Tensor:
        """
        Build the full (N,N) boolean adjacency tensor for the PyTorch model.
        Cross-motif connections are all False (non-edges).                          ## Maybe need to revisit this
        """
        N = self.n_total
        A = torch.zeros(N, N, dtype=torch.bool)
        for slice, motif in zip(self.slices, self.motifs):
            A[slice, slice] = torch.from_numpy(m.adjacency)
        return A

    def summary(self):
        print(f"NetworkSpec: {self.k} motifs, {self.n_total} neurons total")
        for i, motif in enumerate(self.motifs):
            n_edges = motif.adjacency.sum()
            print(f"[{i}] {motif.label}: {motif.n} neurons, "
                  f"{n_edges} edges, "
                  f"eps={motif.eps_init}, delta={motif.delta_init}")
        print(f" delta_cross matrix:\n{np.round(self.cross_deltas, 3)}")


