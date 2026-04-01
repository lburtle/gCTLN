# gctln_w_uniq_edges.py  — v2: fixes for sustained oscillations
# Key changes:
#   1. Non-negative targets (CTLNs can only produce x >= 0)
#   2. Longer training window so decay gets penalized
#   3. Gentler LR schedule
#   4. Theta clamped to safe range during training
#   5. Activity-collapse penalty to prevent fixed-point convergence
#   6. Removed accidental double-training
 
import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt
 
import source
from graph_spec import NetworkSpec, MotifSpec, plot_network_graph
 
 
class GCTLNPerEdge(nn.Module):
    """
    gCTLN with per-edge weight learning, built from a NetworkSpec.
    The NetworkSpec defines the graph topology; this module learns
    the weight magnitudes within the legal gCTLN intervals.
    """
    def __init__(self, spec: NetworkSpec):
        super().__init__()
        self.spec = spec
        self.n    = spec.n_total
 
        # Register full adjacency as a non-learned buffer
        A = spec.to_torch_adjacency()
        self.register_buffer("adjacency", A)
 
        # Per-edge raw parameters — one per entry in W
        # edge_raw[i,j] only used where adjacency[i,j]=True
        # noedge_raw[i,j] only used where adjacency[i,j]=False (and i!=j)
        self.edge_raw   = nn.Parameter(torch.full((self.n, self.n), -2.0))
        self.noedge_raw = nn.Parameter(torch.full((self.n, self.n), -2.0))
 
        # Per-neuron tonic drive — raw param, mapped through softplus
        # so theta is always positive. Initialized to give theta ~ 0.1
        self.theta_raw = nn.Parameter(torch.full((self.n,), -2.25))
 
        # Linear readout
        self.readout = nn.Linear(self.n, self.n, bias=True)
        nn.init.eye_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
 
        # Initialize raw params from spec's eps/delta values
        self._init_from_spec()
 
    @property
    def theta(self):
        """Theta is always positive via softplus, clamped to [0.01, 0.5]."""
        return torch.nn.functional.softplus(self.theta_raw).clamp(0.01, 0.5)
 
    def _init_from_spec(self):
        """
        Warm-start edge_raw and noedge_raw from each motif's
        eps_init and delta_init so the network begins in a
        regime likely to oscillate.
        softplus^{-1}(x) = log(exp(x) - 1)
        """
        def softplus_inv(x):
            return torch.log(torch.expm1(torch.tensor(x).clamp(min=1e-6)))
 
        with torch.no_grad():
            for sl, m in zip(self.spec.slices, self.spec.motifs):
                # edge entries: softplus(edge_raw) = eps_init
                edge_val = softplus_inv(m.eps_init)
                self.edge_raw[sl, sl] = edge_val
 
                # non-edge entries: softplus(noedge_raw) = delta_init
                noedge_val = softplus_inv(m.delta_init)
                self.noedge_raw[sl, sl] = noedge_val
 
    def build_W(self):
        sp = torch.nn.functional.softplus
        A  = self.adjacency.float()
        I  = torch.eye(self.n, device=A.device)
 
        edge_weights   = -1.0 + sp(self.edge_raw)
        noedge_weights = -1.0 - sp(self.noedge_raw)
 
        W = A * edge_weights + (1 - A) * noedge_weights
        W = W * (1 - I)   # zero diagonal
        return W
 
    def ode_func(self, t, x):
        W     = self.build_W()
        drive = W @ x + self.theta
        return -x + torch.clamp(drive, min=0.0)
 
    def forward(self, t_eval, x0):
        traj = odeint(self.ode_func, x0, t_eval,
                      method='dopri5', rtol=1e-4, atol=1e-5)
        pred = self.readout(traj)
        return traj, pred
 
    def limit_cycle_penalty(self):
        W         = self.build_W()
        A         = self.adjacency.float()
        diag_mask = 1 - torch.eye(self.n, device=W.device)
        edge_mean = ((W * A * diag_mask).sum()
                     / (A * diag_mask).sum().clamp(min=1))
        noedge_mask  = (1 - A) * diag_mask
        noedge_mean  = ((W * noedge_mask).sum()
                        / noedge_mask.sum().clamp(min=1))
        violation = torch.clamp(noedge_mean - edge_mean + 0.05, min=0.0)
        return violation ** 2
 
    def activity_penalty(self, traj):
        """
        Penalize collapse: if mean activity in the second half of
        the trajectory drops near zero, add a loss term.
        This encourages the network to stay in an oscillatory regime
        rather than decaying to a fixed point.
        """
        half = traj.shape[0] // 2
        late_mean = traj[half:].mean()
        # We want late_mean to stay above some threshold (e.g. 0.05)
        return torch.clamp(0.05 - late_mean, min=0.0) ** 2
 
    def variance_reward(self, traj):
        """
        Reward temporal variance in the second half of the trajectory.
        Oscillations have high variance; fixed points have zero.
        Returns a penalty (negative reward) that decreases with variance.
        """
        half = traj.shape[0] // 2
        late_var = traj[half:].var(dim=0).mean()
        # We want variance to be at least ~0.01
        return torch.clamp(0.01 - late_var, min=0.0)
 
    def to_numpy_W(self) -> np.ndarray:
        with torch.no_grad():
            return self.build_W().numpy()
 
    def get_edge_summary(self):
        with torch.no_grad():
            W  = self.build_W().numpy()
            sp = torch.nn.functional.softplus
            theta_np = self.theta.numpy()
            print(f"\nWeight matrix W ({self.n}x{self.n}):")
            print(np.round(W, 4))
            print(f"\nTheta: {np.round(theta_np, 4)}")
            print("\nPer-edge weights:")
            for i in range(self.n):
                for j in range(self.n):
                    if self.adjacency[i, j]:
                        eps_ij = sp(self.edge_raw[i, j]).item()
                        print(f"  {j}->{i}: W={W[i,j]:.4f}  eps={eps_ij:.4f}")
            print("\nPer-non-edge weights (sample):")
            shown = 0
            for i in range(self.n):
                for j in range(self.n):
                    if i != j and not self.adjacency[i, j] and shown < 9:
                        delta_ij = sp(self.noedge_raw[i, j]).item()
                        print(f"  {j}->{i}: W={W[i,j]:.4f}  delta={delta_ij:.4f}")
                        shown += 1
 
    def simulate_numpy(self, T: float,
                        n_steps: int, x0: np.ndarray):
        """
        Run a numpy simulation using the learned weights.
        Uses the learned theta values.
        """
        W_np     = self.to_numpy_W()
        theta_np = self.theta.detach().numpy().mean()  # source.py uses scalar
        return source.simulate(W_np, theta_np, T, n_steps, x0)
 
    def visualize_learned(self, T: float = 500, n_steps: int = 15000,
                           transient_frac: float = 0.5,
                           random_seed: int = 42):
        """
        Simulate with learned weights and run the full source.py
        visualization suite on the result.
        """
        np.random.seed(random_seed)
        x0 = np.random.uniform(0.05, 0.4, self.n)
        t, sol = self.simulate_numpy(T, n_steps, x0)
 
        source.full_visualization(
            sol          = sol,
            t            = t,
            slices       = self.spec.slices,
            motif_labels = self.spec.labels,
            transient_frac = transient_frac
        )
        return t, sol
 
    def plot_graph(self, show_weights=True, show_theta=True,
               title="", **kwargs):
        from graph_spec import plot_network_graph
        W_np     = self.to_numpy_W()
        theta_np = self.theta.detach().numpy()
        plot_network_graph(
            self.spec,
            W           = W_np,
            theta       = theta_np,
            show_weights = show_weights,
            show_theta   = show_theta,
            title       = title or "Learned gCTLN weights",
            **kwargs
        )
 
 
def train(model: GCTLNPerEdge,
          targets: torch.Tensor,
          t_span: torch.Tensor,
          x0: torch.Tensor,
          n_epochs: int = 2000,
          lr: float = 3e-3,
          penalty_weight: float = 20.0,
          activity_weight: float = 50.0,
          variance_weight: float = 30.0,
          print_every: int = 200) -> list:
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Gentler decay: step every 800 epochs, gamma=0.5 (was 600, 0.3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=800, gamma=0.5)
 
    loss_history = []
 
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        traj, pred  = model(t_span, x0)
        recon_loss  = nn.functional.mse_loss(pred, targets)
        penalty     = model.limit_cycle_penalty()
        act_pen     = model.activity_penalty(traj)
        var_pen     = model.variance_reward(traj)
        loss        = (recon_loss
                       + penalty_weight  * penalty
                       + activity_weight * act_pen
                       + variance_weight * var_pen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(recon_loss.item())
 
        if epoch % print_every == 0:
            print(f"epoch {epoch:4d} | loss {recon_loss.item():.5f} "
                  f"| penalty {penalty.item():.6f} "
                  f"| act_pen {act_pen.item():.6f} "
                  f"| var_pen {var_pen.item():.6f}")
 
    return loss_history
 
 
if __name__ == "__main__":
 
    # ── 1. Define network topology via NetworkSpec ────────────────────────
    spec = NetworkSpec(
        motifs = [
            MotifSpec.cyclic(3, label="CPG-A",
                             eps_init=0.10, delta_init=0.50),
            MotifSpec.cyclic(3, label="CPG-B",
                             eps_init=0.10, delta_init=0.50),
        ],
        delta_cross = 0.05
    )
    spec.summary()
 
    plot_network_graph(spec, title="Initial network — default weights")
 
    # ── 2. Verify numpy simulation works before training ──────────────────
    W_init = spec.to_numpy_W()
    x0_np  = np.array([0.3, 0.2, 0.1, 0.3, 0.2, 0.1])
    t, sol = source.simulate(W_init, theta=0.1, T=300,
                              n_steps=9000, x0=x0_np)
    cut    = int(0.5 * len(t))
    std_check = sol[cut:].std(axis=0)
    print(f"\nPre-training numpy simulation std: {std_check.round(4)}")
    print("(nonzero = oscillating, as expected)")
 
    # ── 3. Define training target ─────────────────────────────────────────
    # KEY FIX: Non-negative targets!  CTLNs produce x >= 0.
    # Use shifted sinusoids in [0, 1] instead of [-1, 1].
    # Longer window (10*pi ~ 5 full cycles) so decay is penalized.
    t_span       = torch.linspace(0, 10 * np.pi, 1000)
    phase_offset = 2 * np.pi / 3
 
    def nonneg_sin(t, phase=0.0, amp=0.4, offset=0.5):
        """Sinusoid in [offset-amp, offset+amp], entirely non-negative."""
        return offset + amp * torch.sin(t + phase)
 
    targets = torch.stack([
        nonneg_sin(t_span, 0),
        nonneg_sin(t_span, phase_offset),
        nonneg_sin(t_span, 2 * phase_offset),
        nonneg_sin(t_span, 0),                    # CPG-B mirrors CPG-A
        nonneg_sin(t_span, phase_offset),
        nonneg_sin(t_span, 2 * phase_offset),
    ], dim=1)
 
    # ── 4. Train ──────────────────────────────────────────────────────────
    model  = GCTLNPerEdge(spec)
    x0_pt  = torch.tensor([0.15, 0.08, 0.05, 0.15, 0.08, 0.05])
    loss_h = train(model, targets, t_span, x0_pt, n_epochs=2000)
 
    # ── 5. Inspect learned weights ────────────────────────────────────────
    model.get_edge_summary()
    model.plot_graph(title="Learned gCTLN weights")
 
    # ── 6. Plot training curve ────────────────────────────────────────────
    plt.figure(figsize=(8, 3))
    plt.semilogy(loss_h)
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title(f"Training loss — {spec.n_total} neurons, {spec.k} motifs")
    plt.tight_layout()
    plt.savefig("gctln_loss.png", dpi=150)
 
    # ── 7. Visualize learned dynamics using source.py tools ───────────────
    model.visualize_learned(T=500, n_steps=15000)
 
    # ── 8. Compare prediction vs target ──────────────────────────────────
    n_out = spec.n_total
 
    axis_labels = []
    for sl, m in zip(spec.slices, spec.motifs):
        n_neurons = sl.stop - sl.start
        for local_idx in range(n_neurons):
            axis_labels.append(f"{m.label} / n{local_idx}")
 
    with torch.no_grad():
        traj, pred = model(t_span, x0_pt)
 
    fig, axes = plt.subplots(n_out, 1,
                              figsize=(11, 2.5 * n_out),
                              sharex=True)
    if n_out == 1:
        axes = [axes]
 
    motif_palette = ["#c0392b", "#1a5276", "#1e8449",
                     "#7d3c98", "#d35400", "#117a65"]
 
    for i, (ax, lbl) in enumerate(zip(axes, axis_labels)):
        motif_idx = next(mi for mi, sl in enumerate(spec.slices)
                         if sl.start <= i < sl.stop)
        color = motif_palette[motif_idx % len(motif_palette)]
 
        ax.plot(t_span, targets[:, i].numpy(),
                "k--", lw=1.5, label="target")
        ax.plot(t_span, pred[:, i].numpy(),
                color=color, lw=1.5, label="gCTLN")
        ax.plot(t_span, traj[:, i].numpy(),
                color=color, lw=0.8, alpha=0.4, linestyle=":",
                label="raw (pre-readout)")
        ax.set_ylabel(lbl, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
 
    axes[-1].set_xlabel("t")
    plt.suptitle(
        f"gCTLN learned oscillations — "
        f"{spec.k} motifs, {spec.n_total} neurons",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("gctln_fit.png", dpi=150)

## If too many neurons

# ncols  = spec.k                    # one column per motif
# nrows  = max(m.n for m in spec.motifs)   # rows = largest motif size
# fig, axes = plt.subplots(nrows, ncols,
#                           figsize=(5 * ncols, 3 * nrows),
#                           sharex=True)
# if ncols == 1:
#     axes = axes.reshape(-1, 1)

# for motif_idx, (sl, m) in enumerate(zip(spec.slices, spec.motifs)):
#     for local_idx in range(m.n):
#         global_idx = sl.start + local_idx
#         ax = axes[local_idx, motif_idx]
#         color = motif_palette[motif_idx % len(motif_palette)]
#         ax.plot(t_span, targets[:, global_idx].numpy(),
#                 "k--", lw=1.5, label="target")
#         ax.plot(t_span, pred[:, global_idx].numpy(),
#                 color=color, lw=1.5, label="gCTLN")
#         ax.set_title(f"{m.label} / n{local_idx}", fontsize=9)
#         ax.spines[["top", "right"]].set_visible(False)
#         if local_idx == 0:
#             ax.legend(fontsize=7)

#     # Hide unused rows in this column if motifs have different sizes
#     for empty_row in range(m.n, nrows):
#         axes[empty_row, motif_idx].set_visible(False)

# plt.suptitle("gCTLN learned oscillations", fontsize=12)
# plt.tight_layout()
# plt.savefig("gctln_fit.png", dpi=150)