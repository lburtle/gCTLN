# gctln_modular_trainer.py — Modular multi-motif gCTLN training
#
# Two-stage training for arbitrary NetworkSpec configurations:
#   Stage 1 (CMA-ES): Optimize all CTLN parameters (per-edge eps,
#     per-non-edge delta, per-neuron theta, cross-motif deltas)
#     for oscillation structure — frequency, phase, balance.
#   Stage 2 (Gradient descent): Train a readout (linear or MLP)
#     to map raw CTLN waveforms to target signals.
#
# Compatible with graph_spec.py's NetworkSpec and MotifSpec.
# Self-contained simulation — no dependency on source.py.
#
# Usage:
#   from graph_spec import NetworkSpec, MotifSpec
#   spec = NetworkSpec(motifs=[...], delta_cross=0.05)
#   trainer = GCTLNTrainer(spec)
#   trainer.train(targets, t_span, x0)
#   trainer.plot()

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from cmaes import CMA
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import time
import sys

# Import NetworkSpec — fall back to a minimal version if unavailable
try:
    from graph_spec import NetworkSpec, MotifSpec
except ImportError:
    print("Warning: graph_spec.py not found. Using built-in spec classes.")
    from dataclasses import dataclass, field

    @dataclass
    class MotifSpec:
        n: int
        adjacency: np.ndarray
        label: str = ""
        eps_init: float = 0.10
        delta_init: float = 0.50
        def __post_init__(self):
            if not self.label:
                self.label = f"Motif({self.n})"
        @classmethod
        def cyclic(cls, n, label="", **kwargs):
            A = np.zeros((n, n), dtype=bool)
            for j in range(n):
                A[(j + 1) % n, j] = True
            return cls(n=n, adjacency=A, label=label or f"Cycle({n})", **kwargs)

    @dataclass
    class NetworkSpec:
        motifs: list
        delta_cross: float = 0.05
        cross_deltas: Optional[np.ndarray] = None
        def __post_init__(self):
            k = len(self.motifs)
            if self.cross_deltas is None:
                self.cross_deltas = np.full((k, k), self.delta_cross)
            np.fill_diagonal(self.cross_deltas, 0.0)
        @property
        def n_total(self): return sum(m.n for m in self.motifs)
        @property
        def k(self): return len(self.motifs)
        @property
        def slices(self):
            starts = np.concatenate(([0], np.cumsum([m.n for m in self.motifs])[:-1])).astype(int)
            return [slice(s, s + m.n) for s, m in zip(starts, self.motifs)]
        @property
        def labels(self): return [m.label for m in self.motifs]
        def summary(self):
            print(f"NetworkSpec: {self.k} motifs, {self.n_total} neurons total")
            for i, m in enumerate(self.motifs):
                print(f"  [{i}] {m.label}: {m.n}n, eps={m.eps_init}, delta={m.delta_init}")
            print(f"  cross_deltas:\n{np.round(self.cross_deltas, 3)}")


# ═══════════════════════════════════════════════════════════════════════
# 1. CTLN simulation from NetworkSpec
# ═══════════════════════════════════════════════════════════════════════

def build_W_from_params(spec: NetworkSpec,
                        eps_all: np.ndarray,
                        delta_all: np.ndarray,
                        cross_deltas: np.ndarray) -> np.ndarray:
    """
    Build the full weight matrix from per-edge parameters.

    Parameters
    ----------
    spec         : NetworkSpec defining graph topology
    eps_all      : flat array of all edge eps values (one per edge in the graph)
    delta_all    : flat array of all non-edge delta values
    cross_deltas : (k, k) array of cross-motif delta values

    Returns
    -------
    W : (N, N) weight matrix
    """
    N = spec.n_total
    W = np.zeros((N, N))
    slices = spec.slices

    eps_idx = 0
    delta_idx = 0

    for i, (si, mi) in enumerate(zip(slices, spec.motifs)):
        for j, (sj, mj) in enumerate(zip(slices, spec.motifs)):
            if i == j:
                # Within-motif: per-edge eps and delta
                for r in range(mi.n):
                    for c in range(mi.n):
                        if r == c:
                            W[si.start + r, si.start + c] = 0.0
                        elif mi.adjacency[r, c]:
                            W[si.start + r, si.start + c] = -1.0 + eps_all[eps_idx]
                            eps_idx += 1
                        else:
                            W[si.start + r, si.start + c] = -1.0 - delta_all[delta_idx]
                            delta_idx += 1
            else:
                # Cross-motif: uniform delta from cross_deltas matrix
                dc = cross_deltas[i, j]
                for r in range(mi.n):
                    for c in range(mj.n):
                        W[si.start + r, sj.start + c] = -1.0 - dc
    return W


def count_params(spec: NetworkSpec) -> dict:
    """Count the number of each parameter type for a NetworkSpec."""
    n_eps = 0
    n_delta_within = 0
    for m in spec.motifs:
        n_edges = int(m.adjacency.sum())
        n_nonedges = m.n * (m.n - 1) - n_edges
        n_eps += n_edges
        n_delta_within += n_nonedges

    n_theta = spec.n_total
    k = spec.k
    n_cross = k * (k - 1) // 2  # upper triangle of cross_deltas

    total = n_eps + n_delta_within + n_theta + n_cross
    return {
        'n_eps': n_eps,
        'n_delta_within': n_delta_within,
        'n_theta': n_theta,
        'n_cross': n_cross,
        'total': total,
    }


def dxdt_ctln(x, t, W, theta):
    """CTLN dynamics: dx/dt = -x + [Wx + theta]_+"""
    return -x + np.maximum(W @ x + theta, 0)


def simulate(W, theta, T, n_steps, x0):
    """Run CTLN simulation."""
    t = np.linspace(0, T, n_steps)
    sol = odeint(dxdt_ctln, x0, t, args=(W, theta),
                 mxstep=10000)
    return t, sol


# ═══════════════════════════════════════════════════════════════════════
# 2. Parameter encoding / decoding
# ═══════════════════════════════════════════════════════════════════════

class ParamCodec:
    """
    Encodes/decodes between an unconstrained CMA-ES vector and
    legal gCTLN parameters for a given NetworkSpec.

    Layout of raw vector:
      [0 : n_eps]                         -> per-edge eps values
      [n_eps : n_eps+n_delta]             -> per-non-edge delta values
      [n_eps+n_delta : n_eps+n_delta+N]   -> per-neuron theta values
      [... : end]                         -> cross-motif deltas (upper tri)
    """
    def __init__(self, spec: NetworkSpec):
        self.spec = spec
        self.counts = count_params(spec)
        self.dim = self.counts['total']

    def decode(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
        """
        Returns: eps_all, delta_all, theta, cross_deltas
        """
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        def softplus(x):
            return np.log1p(np.exp(np.clip(x, -20, 20)))

        c = self.counts
        idx = 0

        # Per-edge eps: in (0.01, 0.99)
        eps_all = 0.01 + 0.98 * sigmoid(raw[idx:idx + c['n_eps']])
        idx += c['n_eps']

        # Per-non-edge delta: in (0.01, inf)
        delta_all = 0.01 + softplus(raw[idx:idx + c['n_delta_within']])
        idx += c['n_delta_within']

        # Per-neuron theta: in (0.01, 2.0)
        theta = 0.01 + 1.99 * sigmoid(raw[idx:idx + c['n_theta']])
        idx += c['n_theta']

        # Cross-motif deltas: in (0.01, inf)
        k = self.spec.k
        cross_flat = 0.01 + softplus(raw[idx:idx + c['n_cross']])
        cross_deltas = np.zeros((k, k))
        flat_idx = 0
        for i in range(k):
            for j in range(i + 1, k):
                cross_deltas[i, j] = cross_flat[flat_idx]
                cross_deltas[j, i] = cross_flat[flat_idx]
                flat_idx += 1

        return eps_all, delta_all, theta, cross_deltas

    def encode(self, eps_all, delta_all, theta, cross_deltas) -> np.ndarray:
        """Inverse of decode — for initialization."""
        def sigmoid_inv(y):
            y = np.clip(y, 1e-6, 1 - 1e-6)
            return np.log(y / (1 - y))
        def softplus_inv(y):
            return np.log(np.expm1(np.clip(y, 1e-6, 50)))

        c = self.counts
        raw = np.zeros(self.dim)
        idx = 0

        raw[idx:idx + c['n_eps']] = sigmoid_inv((eps_all - 0.01) / 0.98)
        idx += c['n_eps']

        raw[idx:idx + c['n_delta_within']] = softplus_inv(delta_all - 0.01)
        idx += c['n_delta_within']

        raw[idx:idx + c['n_theta']] = sigmoid_inv((theta - 0.01) / 1.99)
        idx += c['n_theta']

        k = self.spec.k
        flat_idx = 0
        for i in range(k):
            for j in range(i + 1, k):
                raw[idx + flat_idx] = softplus_inv(cross_deltas[i, j] - 0.01)
                flat_idx += 1

        return raw

    def init_from_spec(self) -> np.ndarray:
        """Create initial raw vector from spec's default parameters."""
        c = self.counts

        # All edges start at their motif's eps_init
        eps_all = []
        delta_all = []
        for m in self.spec.motifs:
            for r in range(m.n):
                for col in range(m.n):
                    if r != col:
                        if m.adjacency[r, col]:
                            eps_all.append(m.eps_init)
                        else:
                            delta_all.append(m.delta_init)
        eps_all = np.array(eps_all)
        delta_all = np.array(delta_all)

        theta = np.full(self.spec.n_total, 0.1)

        cross_deltas = self.spec.cross_deltas.copy()

        return self.encode(eps_all, delta_all, theta, cross_deltas)

    def build_W(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode raw vector and build W matrix. Returns (W, theta)."""
        eps_all, delta_all, theta, cross_deltas = self.decode(raw)
        W = build_W_from_params(self.spec, eps_all, delta_all, cross_deltas)
        return W, theta


# ═══════════════════════════════════════════════════════════════════════
# 3. Stage 1: CMA-ES for oscillation structure
# ═══════════════════════════════════════════════════════════════════════

def estimate_frequency(signal, dt):
    """Estimate oscillation frequency from peak spacing."""
    signal = signal - signal.mean()
    std = signal.std()
    if std < 1e-8:
        return 0.0
    peaks, _ = find_peaks(signal, height=0.01 * std,
                          distance=max(1, int(0.3 / dt)))
    if len(peaks) < 2:
        return 0.0
    periods = np.diff(peaks) * dt
    return 1.0 / periods.mean()


@dataclass
class Stage1Config:
    """Configuration for Stage 1 CMA-ES optimization."""
    target_freq:     float = 0.1592   # target oscillation frequency
    T:               float = 30 * np.pi  # simulation time (long for freq stability)
    n_steps:         int   = 3000
    n_generations:   int   = 400
    population_size: int   = 50
    sigma0:          float = 0.5
    print_every:     int   = 25
    # Loss weights
    w_freq:     float = 10.0   # frequency match
    w_phase:    float = 1.0    # phase relationships
    w_variance: float = 0.5    # reward oscillation amplitude
    w_balance:  float = 0.3    # reward balanced activation across neurons
    # Target phase offsets between motifs (None = don't constrain)
    # e.g. [np.pi] for anti-phase between two motifs
    inter_motif_phase_targets: Optional[List[float]] = None


def stage1_loss(raw_params, codec: ParamCodec, config: Stage1Config,
                x0: np.ndarray) -> float:
    """
    Evaluate oscillation quality for CMA-ES.
    Rewards: correct frequency, balanced activation, sequential phases.
    Does NOT match waveform shape.
    """
    try:
        W, theta = codec.build_W(raw_params)
        t, sol = simulate(W, theta, T=config.T, n_steps=config.n_steps, x0=x0)

        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)) or sol.max() > 100:
            return 1e6

        dt = t[1] - t[0]
        half = len(t) // 2
        late = sol[half:]

        # ── Variance: all neurons should oscillate ────────────────────
        per_neuron_var = late.var(axis=0)
        mean_var = per_neuron_var.mean()
        if mean_var < 1e-4:
            return 1e6

        min_var = per_neuron_var.min()
        var_balance = min_var / (mean_var + 1e-8)

        # ── Frequency match per motif ─────────────────────────────────
        spec = codec.spec
        freq_errors = []
        motif_freqs = []
        for sl in spec.slices:
            # Use the first neuron of each motif as reference
            freq = estimate_frequency(late[:, sl.start], dt)
            if freq == 0:
                return 1e6
            motif_freqs.append(freq)
            freq_errors.append((freq - config.target_freq) ** 2)
        freq_loss = np.mean(freq_errors)

        # ── Within-motif phase: sequential activation ─────────────────
        phase_loss = 0.0
        for sl, m in zip(spec.slices, spec.motifs):
            n = m.n
            if n < 2:
                continue
            expected_offset = 1.0 / (freq * n) if motif_freqs else 0
            neuron_peaks = []
            for ni in range(sl.start, sl.stop):
                sig = late[:, ni] - late[:, ni].mean()
                peaks, _ = find_peaks(sig, distance=max(1, int(0.3 / dt)))
                if len(peaks) >= 2:
                    neuron_peaks.append(peaks)
                else:
                    neuron_peaks.append(None)
            # Check consecutive neuron phase offsets
            for ni in range(len(neuron_peaks) - 1):
                if neuron_peaks[ni] is not None and neuron_peaks[ni+1] is not None:
                    n_comp = min(len(neuron_peaks[ni]), len(neuron_peaks[ni+1]))
                    if n_comp > 0:
                        diffs = (neuron_peaks[ni+1][:n_comp] -
                                 neuron_peaks[ni][:n_comp]) * dt
                        phase_loss += (np.mean(np.abs(diffs)) - expected_offset) ** 2

        # ── Inter-motif phase (if specified) ──────────────────────────
        inter_phase_loss = 0.0
        if (config.inter_motif_phase_targets is not None
                and len(spec.slices) >= 2):
            pair_idx = 0
            for i in range(spec.k):
                for j in range(i + 1, spec.k):
                    if pair_idx >= len(config.inter_motif_phase_targets):
                        break
                    target_phase = config.inter_motif_phase_targets[pair_idx]
                    # Measure phase difference between first neurons of each motif
                    peaks_i, _ = find_peaks(
                        late[:, spec.slices[i].start] -
                        late[:, spec.slices[i].start].mean(),
                        distance=max(1, int(0.3 / dt)))
                    peaks_j, _ = find_peaks(
                        late[:, spec.slices[j].start] -
                        late[:, spec.slices[j].start].mean(),
                        distance=max(1, int(0.3 / dt)))
                    if len(peaks_i) >= 2 and len(peaks_j) >= 2:
                        n_comp = min(len(peaks_i), len(peaks_j))
                        diffs = (peaks_j[:n_comp] - peaks_i[:n_comp]) * dt
                        mean_period = 1.0 / config.target_freq
                        # Target phase in time units
                        target_dt = target_phase / (2 * np.pi) * mean_period
                        inter_phase_loss += (np.mean(np.abs(diffs)) - target_dt) ** 2
                    else:
                        inter_phase_loss += 0.1
                    pair_idx += 1

        # ── Combined loss ─────────────────────────────────────────────
        loss = (config.w_freq * freq_loss
                + config.w_phase * (phase_loss + inter_phase_loss)
                - config.w_variance * mean_var
                - config.w_balance * var_balance)

        return loss

    except Exception as e:
        return 1e6


def train_stage1(spec: NetworkSpec, x0: np.ndarray,
                 config: Stage1Config) -> Tuple[np.ndarray, list]:
    """
    Stage 1: CMA-ES optimization for oscillation structure.

    Returns
    -------
    best_params : raw parameter vector
    history     : list of (generation, best_loss)
    """
    codec = ParamCodec(spec)
    counts = codec.counts

    print("=" * 65)
    print("STAGE 1: CMA-ES for oscillation structure")
    print(f"  Network: {spec.k} motifs, {spec.n_total} neurons")
    print(f"  Parameters: {codec.dim} total "
          f"({counts['n_eps']} eps, {counts['n_delta_within']} delta, "
          f"{counts['n_theta']} theta, {counts['n_cross']} cross)")
    print(f"  Target freq: {config.target_freq:.4f} Hz")
    print(f"  Population: {config.population_size}, "
          f"Generations: {config.n_generations}")
    if config.inter_motif_phase_targets:
        print(f"  Inter-motif phase targets: "
              f"{[f'{p:.2f} rad' for p in config.inter_motif_phase_targets]}")
    print("=" * 65)

    raw_init = codec.init_from_spec()

    optimizer = CMA(mean=raw_init, sigma=config.sigma0,
                    population_size=config.population_size)

    best_loss = float('inf')
    best_params = raw_init.copy()
    history = []
    start = time.time()

    for gen in range(config.n_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            params = optimizer.ask()
            loss = stage1_loss(params, codec, config, x0)
            solutions.append((params, loss))

        optimizer.tell(solutions)

        gen_best = min(solutions, key=lambda s: s[1])
        if gen_best[1] < best_loss:
            best_loss = gen_best[1]
            best_params = gen_best[0].copy()

        history.append((gen, best_loss))

        if gen % config.print_every == 0:
            W, theta = codec.build_W(best_params)
            # Estimate freq from first motif's first neuron
            t_check, sol_check = simulate(W, theta, T=config.T,
                                          n_steps=config.n_steps, x0=x0)
            dt = t_check[1] - t_check[0]
            half = len(t_check) // 2
            freq = estimate_frequency(sol_check[half:, 0], dt)
            elapsed = time.time() - start
            print(f"  gen {gen:4d} | loss {best_loss:+.4f} "
                  f"| freq {freq:.4f} "
                  f"| {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"\n  Stage 1 done in {elapsed:.1f}s. Best loss: {best_loss:.4f}")
    return best_params, history


# ═══════════════════════════════════════════════════════════════════════
# 4. Stage 2: Readout training (linear or MLP)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Stage2Config:
    """Configuration for Stage 2 readout training."""
    n_epochs:    int   = 3000
    lr:          float = 1e-2
    readout_type: Literal['linear', 'mlp'] = 'linear'
    mlp_hidden:  int   = 32
    print_every: int   = 300


def make_readout(n_in: int, n_out: int, config: Stage2Config) -> nn.Module:
    """Create the readout module."""
    if config.readout_type == 'linear':
        layer = nn.Linear(n_in, n_out)
        # Initialize close to identity if dimensions match
        if n_in == n_out:
            nn.init.eye_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer
    else:
        return nn.Sequential(
            nn.Linear(n_in, config.mlp_hidden),
            nn.SiLU(),
            nn.Linear(config.mlp_hidden, config.mlp_hidden),
            nn.SiLU(),
            nn.Linear(config.mlp_hidden, n_out),
        )


def train_stage2(spec: NetworkSpec,
                 ctln_params: np.ndarray,
                 t_span: np.ndarray,
                 targets: np.ndarray,
                 x0: np.ndarray,
                 config: Stage2Config,
                 n_out: Optional[int] = None,
                 ) -> Tuple[nn.Module, np.ndarray, np.ndarray, list]:
    """
    Stage 2: Train readout to map frozen CTLN trajectory to targets.

    Parameters
    ----------
    n_out : output dimension. If None, uses targets.shape[1].
            Can differ from n_total if targets have fewer channels
            (e.g. only readout limb neurons, not auxiliary neurons).
    """
    codec = ParamCodec(spec)
    W, theta = codec.build_W(ctln_params)
    N = spec.n_total
    if n_out is None:
        n_out = targets.shape[1]

    print("\n" + "=" * 65)
    print("STAGE 2: Readout training")
    print(f"  Type: {config.readout_type}")
    print(f"  Input dim: {N} (CTLN neurons)")
    print(f"  Output dim: {n_out}")
    print("=" * 65)

    # Generate frozen CTLN trajectory
    t, sol = simulate(W, theta, T=t_span[-1], n_steps=len(t_span), x0=x0)
    ctln_traj = torch.from_numpy(sol).float()
    target_tensor = torch.from_numpy(targets).float()

    # Build readout
    readout = make_readout(N, n_out, config)
    n_readout_params = sum(p.numel() for p in readout.parameters())
    print(f"  Readout params: {n_readout_params}")

    optimizer = torch.optim.Adam(readout.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.3)

    loss_history = []
    for epoch in range(config.n_epochs):
        optimizer.zero_grad()
        pred = readout(ctln_traj)
        loss = nn.functional.mse_loss(pred, target_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch % config.print_every == 0:
            print(f"  epoch {epoch:4d} | MSE {loss.item():.6f}")

    print(f"\n  Stage 2 done. Final MSE: {loss_history[-1]:.6f}")

    with torch.no_grad():
        final_pred = readout(ctln_traj).numpy()

    return readout, final_pred, sol, loss_history


# ═══════════════════════════════════════════════════════════════════════
# 5. High-level trainer class
# ═══════════════════════════════════════════════════════════════════════

class GCTLNTrainer:
    """
    High-level trainer wrapping both stages.

    Usage:
        spec = NetworkSpec(motifs=[...])
        trainer = GCTLNTrainer(spec)
        trainer.train(targets, t_span, x0)
        trainer.plot()
        trainer.summary()
    """
    def __init__(self, spec: NetworkSpec,
                 stage1_config: Optional[Stage1Config] = None,
                 stage2_config: Optional[Stage2Config] = None):
        self.spec = spec
        self.codec = ParamCodec(spec)
        self.s1_config = stage1_config or Stage1Config()
        self.s2_config = stage2_config or Stage2Config()

        # Filled after training
        self.ctln_params = None
        self.readout = None
        self.ctln_traj = None
        self.final_pred = None
        self.targets = None
        self.t_span = None
        self.s1_history = None
        self.s2_history = None

    def train(self, targets: np.ndarray, t_span: np.ndarray,
              x0: np.ndarray, n_out: Optional[int] = None):
        """Run both training stages."""
        self.targets = targets
        self.t_span = t_span

        self.spec.summary()
        print(f"CMA-ES parameters: {self.codec.dim}")
        print()

        # Stage 1
        self.ctln_params, self.s1_history = train_stage1(
            self.spec, x0, self.s1_config)

        # Print learned CTLN parameters
        W, theta = self.codec.build_W(self.ctln_params)
        print(f"\n  Learned W:\n{np.round(W, 4)}")
        print(f"  Learned theta: {np.round(theta, 4)}")

        # Verify oscillation
        t_check, sol_check = simulate(W, theta, T=200, n_steps=6000, x0=x0)
        cut = len(t_check) // 2
        dt = t_check[1] - t_check[0]
        for i, (sl, label) in enumerate(zip(self.spec.slices, self.spec.labels)):
            std = sol_check[cut:, sl].std(axis=0).mean()
            freq = estimate_frequency(sol_check[cut:, sl.start], dt)
            tag = "oscillating" if std > 0.01 else "STATIC"
            print(f"  {label}: std={std:.4f} freq={freq:.4f} [{tag}]")

        # Stage 2
        self.readout, self.final_pred, self.ctln_traj, self.s2_history = \
            train_stage2(self.spec, self.ctln_params, t_span, targets,
                         x0, self.s2_config, n_out=n_out)

    def summary(self):
        """Print final results."""
        if self.final_pred is None:
            print("Not trained yet. Call .train() first.")
            return
        mse = np.mean((self.final_pred - self.targets) ** 2)
        n_readout = sum(p.numel() for p in self.readout.parameters())
        print(f"\n{'='*65}")
        print(f"FINAL RESULTS")
        print(f"{'='*65}")
        print(f"  Overall MSE: {mse:.6f}")
        print(f"  CTLN params: {self.codec.dim} (frozen after stage 1)")
        print(f"  Readout params: {n_readout}")
        print(f"  Total params: {self.codec.dim + n_readout}")

    def plot(self, prefix="gctln_modular"):
        """Generate all plots."""
        if self.final_pred is None:
            print("Not trained yet.")
            return

        spec = self.spec
        t = self.t_span
        targets = self.targets
        ctln = self.ctln_traj
        pred = self.final_pred
        n_out = targets.shape[1]

        # Color palette
        base_colors = ["#c0392b", "#1a5276", "#1e8449",
                       "#7d3c98", "#d35400", "#117a65",
                       "#884ea0", "#148f77"]

        # ── Fit plot ──────────────────────────────────────────────────
        fig, axes = plt.subplots(n_out, 1,
                                  figsize=(12, 2.5 * n_out),
                                  sharex=True)
        if n_out == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            color = base_colors[i % len(base_colors)]
            # Find which motif this output belongs to
            label = f"output {i}"
            for mi, sl in enumerate(spec.slices):
                if i < spec.n_total and sl.start <= i < sl.stop:
                    label = f"{spec.labels[mi]} / n{i - sl.start}"
                    break

            ax.plot(t, targets[:, i], "k--", lw=1.5, label="target")
            if i < ctln.shape[1]:
                ax.plot(t, ctln[:, i], color=color, lw=0.8, alpha=0.3,
                        linestyle=":", label="raw CTLN")
            ax.plot(t, pred[:, i], color=color, lw=1.5,
                    label="readout output")
            ax.set_ylabel(label, fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            ax.spines[["top", "right"]].set_visible(False)

        axes[-1].set_xlabel("t")
        axes[0].set_title(
            f"gCTLN fit — {spec.k} motifs, {spec.n_total} neurons",
            fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{prefix}_fit.png", dpi=150)
        plt.close()

        # ── Training curves ───────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

        gens, losses = zip(*self.s1_history)
        ax1.plot(gens, losses)
        ax1.set_xlabel("generation")
        ax1.set_ylabel("stage 1 loss")
        ax1.set_title("Stage 1: CMA-ES")
        ax1.spines[["top", "right"]].set_visible(False)

        ax2.semilogy(self.s2_history)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("MSE")
        ax2.set_title("Stage 2: readout")
        ax2.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"{prefix}_loss.png", dpi=150)
        plt.close()

        # ── Raw CTLN waveforms ────────────────────────────────────────
        N = spec.n_total
        cut = int(0.6 * len(t))
        fig, axes = plt.subplots(N, 1, figsize=(12, 2 * N), sharex=True)
        if N == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            mi = next(m for m, sl in enumerate(spec.slices)
                      if sl.start <= i < sl.stop)
            color = base_colors[mi % len(base_colors)]
            ax.plot(t[cut:], ctln[cut:, i], color=color, lw=1.2)
            ax.set_ylabel(f"{spec.labels[mi]}/n{i - spec.slices[mi].start}",
                          fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)

        axes[-1].set_xlabel("t")
        axes[0].set_title("Raw CTLN firing rates (late trajectory)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{prefix}_raw.png", dpi=150)
        plt.close()

        print(f"\nSaved: {prefix}_fit.png, {prefix}_loss.png, {prefix}_raw.png")


# ═══════════════════════════════════════════════════════════════════════
# 6. Example configurations
# ═══════════════════════════════════════════════════════════════════════

def example_single_motif():
    """Single 3-cycle matching sinusoidal target."""
    spec = NetworkSpec(
        motifs=[MotifSpec.cyclic(3, label="CPG-A",
                                 eps_init=0.10, delta_init=0.50)])

    T = 10 * np.pi
    t_span = np.linspace(0, T, 1000)
    phase_offset = 2 * np.pi / 3
    targets = np.stack([
        0.5 + 0.4 * np.sin(t_span),
        0.5 + 0.4 * np.sin(t_span + phase_offset),
        0.5 + 0.4 * np.sin(t_span + 2 * phase_offset),
    ], axis=1)
    x0 = np.array([0.15, 0.08, 0.05])

    trainer = GCTLNTrainer(
        spec,
        stage1_config=Stage1Config(target_freq=1/(2*np.pi)),
        stage2_config=Stage2Config(readout_type='linear'))
    trainer.train(targets, t_span, x0)
    trainer.summary()
    trainer.plot(prefix="single_motif")


def example_anti_phase_pair():
    """Two anti-phase 3-cycles — vortex shedding analog."""
    spec = NetworkSpec(
        motifs=[
            MotifSpec.cyclic(3, label="Upper", eps_init=0.10, delta_init=0.50),
            MotifSpec.cyclic(3, label="Lower", eps_init=0.10, delta_init=0.50),
        ],
        delta_cross=0.5)

    T = 10 * np.pi
    t_span = np.linspace(0, T, 1000)
    freq = 1 / (2 * np.pi)
    phase_offset = 2 * np.pi / 3

    # Upper motif: sin with 120-deg offsets
    # Lower motif: same but shifted by pi (anti-phase)
    targets = np.stack([
        0.5 + 0.4 * np.sin(t_span),
        0.5 + 0.4 * np.sin(t_span + phase_offset),
        0.5 + 0.4 * np.sin(t_span + 2 * phase_offset),
        0.5 + 0.4 * np.sin(t_span + np.pi),
        0.5 + 0.4 * np.sin(t_span + np.pi + phase_offset),
        0.5 + 0.4 * np.sin(t_span + np.pi + 2 * phase_offset),
    ], axis=1)
    x0 = np.array([0.15, 0.08, 0.05, 0.05, 0.15, 0.08])

    trainer = GCTLNTrainer(
        spec,
        stage1_config=Stage1Config(
            target_freq=freq,
            inter_motif_phase_targets=[np.pi],  # anti-phase
            n_generations=500,
            population_size=60),
        stage2_config=Stage2Config(readout_type='linear'))
    trainer.train(targets, t_span, x0)
    trainer.summary()
    trainer.plot(prefix="anti_phase")


def example_mlp_readout():
    """Single motif with MLP readout for richer waveform shaping."""
    spec = NetworkSpec(
        motifs=[MotifSpec.cyclic(3, label="CPG-A",
                                 eps_init=0.10, delta_init=0.50)])

    T = 10 * np.pi
    t_span = np.linspace(0, T, 1000)
    phase_offset = 2 * np.pi / 3
    targets = np.stack([
        0.5 + 0.4 * np.sin(t_span),
        0.5 + 0.4 * np.sin(t_span + phase_offset),
        0.5 + 0.4 * np.sin(t_span + 2 * phase_offset),
    ], axis=1)
    x0 = np.array([0.15, 0.08, 0.05])

    trainer = GCTLNTrainer(
        spec,
        stage1_config=Stage1Config(target_freq=1/(2*np.pi)),
        stage2_config=Stage2Config(readout_type='mlp', mlp_hidden=32))
    trainer.train(targets, t_span, x0)
    trainer.summary()
    trainer.plot(prefix="mlp_readout")


# ═══════════════════════════════════════════════════════════════════════
# 7. Main — run from command line
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    examples = {
        'single': example_single_motif,
        'antiphase': example_anti_phase_pair,
        'mlp': example_mlp_readout,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Usage: python gctln_modular_trainer.py [single|antiphase|mlp]")
        print("\nExamples:")
        print("  python gctln_modular_trainer.py single     # one 3-cycle")
        print("  python gctln_modular_trainer.py antiphase  # two anti-phase 3-cycles")
        print("  python gctln_modular_trainer.py mlp        # MLP readout")
        print("\nRunning 'single' by default...\n")
        example_single_motif()