
# gnn_sine_baseline.py
#
# Standard GNN baseline for the 3-sine-wave toy problem used in gCTLN work.
# The goal: show that a vanilla GNN struggles to model sustained oscillations,
# providing a contrast to the gCTLN's native attractor dynamics.
#
# Target: three non-negative sines at 120° phase offsets:
#   x_i(t) = 0.4 * sin(t + i * 2π/3) + 0.5,   i = 0, 1, 2
#
# Graph: 3 nodes in a 3-cycle (matches the single CPG-A motif topology
# used in your gCTLN experiments).
#
# Two baselines are trained:
#   (1) Feedforward GNN  : input = [t, node_id_onehot], output = x_i(t)
#   (2) Recurrent GNN    : x(t+dt) = x(t) + dt * f_GNN(x(t)); rollout from x(0)
#
# The feedforward model can in principle memorize the sine as a function of t
# but has no dynamical structure; the recurrent model is the fair comparison
# to an ODE-based CTLN, and is the one that exposes the real gap.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)

OUT_DIR = Path("/scratch/xfd3tf/gCTLN/imgs")
OUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. Target signal (matches gCTLN experiments exactly)
# ═══════════════════════════════════════════════════════════════════════

def nonneg_sin(t: torch.Tensor, phase: float,
               amp: float = 0.4, offset: float = 0.5) -> torch.Tensor:
    """Non-negative sine: values in [offset-amp, offset+amp]."""
    return amp * torch.sin(t + phase) + offset


N_NODES = 3
PHASE_OFFSET = 2 * np.pi / 3  # 120°
T_MAX = 10 * np.pi            # ~5 full periods
N_STEPS = 1000

t_train = torch.linspace(0.0, T_MAX, N_STEPS)                      # (T,)
targets = torch.stack([                                            # (T, 3)
    nonneg_sin(t_train, 0.0),
    nonneg_sin(t_train, PHASE_OFFSET),
    nonneg_sin(t_train, 2 * PHASE_OFFSET),
], dim=1)

# Graph: directed 3-cycle 0 -> 1 -> 2 -> 0, same as the CPG-A motif
edge_index = torch.tensor([[0, 1, 2],
                           [1, 2, 0]], dtype=torch.long)   # (2, E)


# ═══════════════════════════════════════════════════════════════════════
# 2. Simple message-passing layer (no torch_geometric dependency)
# ═══════════════════════════════════════════════════════════════════════

class MPLayer(nn.Module):
    """
    Standard message-passing layer:
        m_ji = MLP_msg([h_j, h_i])
        agg_i = sum_j m_ji
        h_i' = MLP_upd([h_i, agg_i])
    Shared weights across all edges and all nodes — the defining GNN property.
    """
    def __init__(self, d_h: int, d_msg: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * d_h, d_msg), nn.SiLU(),
            nn.Linear(d_msg, d_msg),
        )
        self.upd = nn.Sequential(
            nn.Linear(d_h + d_msg, d_h), nn.SiLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]                       # (E,)
        m = self.msg(torch.cat([h[src], h[dst]], dim=-1))             # (E, d_msg)
        agg = torch.zeros(h.size(0), m.size(-1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, m)                                     # sum aggregation
        return self.upd(torch.cat([h, agg], dim=-1))                  # (N, d_h)


# ═══════════════════════════════════════════════════════════════════════
# 3. Feedforward GNN — takes t as input, predicts x_i(t)
# ═══════════════════════════════════════════════════════════════════════

class FeedforwardGNN(nn.Module):
    """
    For each time t, build node features = [t, one_hot_node_id], run K
    message-passing rounds, then readout to a scalar per node.
    No temporal recurrence — t is just an input coordinate.
    """
    def __init__(self, n_nodes: int = 3, d_h: int = 32, d_msg: int = 32, n_layers: int = 3):
        super().__init__()
        self.n_nodes = n_nodes
        self.encode = nn.Linear(1 + n_nodes, d_h)
        self.layers = nn.ModuleList([MPLayer(d_h, d_msg) for _ in range(n_layers)])
        self.readout = nn.Linear(d_h, 1)
        self.register_buffer("node_onehot", torch.eye(n_nodes))

    def forward(self, t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # t: (T,)  ->  out: (T, N)
        T = t.size(0)
        N = self.n_nodes
        t_feat = t.view(T, 1, 1).expand(T, N, 1)                       # (T, N, 1)
        id_feat = self.node_onehot.unsqueeze(0).expand(T, N, N)        # (T, N, N)
        x = torch.cat([t_feat, id_feat], dim=-1)                       # (T, N, 1+N)

        # process each timestep independently (trivially parallel)
        x = x.reshape(T * N, -1)                                       # (T*N, 1+N)
        h = self.encode(x)
        # build a batched edge_index for T disjoint copies of the 3-cycle
        offsets = (torch.arange(T, device=t.device) * N).view(T, 1, 1) # (T,1,1)
        ei = edge_index.unsqueeze(0) + offsets                         # (T, 2, E)
        ei = ei.permute(1, 0, 2).reshape(2, -1)                        # (2, T*E)
        for layer in self.layers:
            h = h + layer(h, ei)                                       # residual
        out = self.readout(h).view(T, N)                               # (T, N)
        return out


# ═══════════════════════════════════════════════════════════════════════
# 4. Recurrent GNN — integrates forward via learned message passing
# ═══════════════════════════════════════════════════════════════════════

class RecurrentGNN(nn.Module):
    """
    Each node carries a hidden state h_i ∈ R^{d_h} that evolves as
        h(t+dt) = h(t) + dt * GNN_step(h(t))
    i.e. a learned neural ODE where the vector field is a GNN. A linear
    readout maps h_i -> x_i. Rolled out from a learned initial state h(0).
    This is the fair comparison to a CTLN ODE: both are recurrent
    dynamical systems on the same graph.
    """
    def __init__(self, n_nodes: int = 3, d_h: int = 16, d_msg: int = 32, n_layers: int = 2):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_h = d_h
        self.layers = nn.ModuleList([MPLayer(d_h, d_msg) for _ in range(n_layers)])
        self.readout = nn.Linear(d_h, 1)
        # learned initial state per node
        self.h0 = nn.Parameter(0.1 * torch.randn(n_nodes, d_h))

    def vector_field(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = h
        for layer in self.layers:
            out = out + layer(out, edge_index)                         # residual
        return out - h  # dh/dt = GNN(h) - h  (decay term stabilizes)

    def forward(self, t: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Euler integration on the time grid t
        T = t.size(0)
        dts = torch.diff(t, prepend=t[:1])                             # (T,) with dt[0] = 0
        h = self.h0
        outs = []
        for k in range(T):
            if k > 0:
                h = h + dts[k] * self.vector_field(h, edge_index)
            outs.append(self.readout(h).squeeze(-1))                   # (N,)
        return torch.stack(outs, dim=0)                                # (T, N)


# ═══════════════════════════════════════════════════════════════════════
# 5. Training loops
# ═══════════════════════════════════════════════════════════════════════

def train(model, name, n_epochs=2000, lr=3e-3, log_every=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        pred = model(t_train, edge_index)                              # (T, 3)
        loss = F.mse_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        history.append(loss.item())
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"  [{name}] epoch {epoch:4d}  loss = {loss.item():.5f}")
    return history


# ═══════════════════════════════════════════════════════════════════════
# 6. Evaluation: in-range fit + extrapolation test
# ═══════════════════════════════════════════════════════════════════════

def plot_fit(model, name, history, feedforward: bool):
    """Save two figures: (a) fit + loss, (b) extrapolation."""
    model.eval()
    with torch.no_grad():
        # in-range prediction
        pred = model(t_train, edge_index).cpu().numpy()
        # extrapolation: roll out/evaluate to 2x the training horizon
        t_ext = torch.linspace(0.0, 2 * T_MAX, 2 * N_STEPS)
        pred_ext = model(t_ext, edge_index).cpu().numpy()
        target_ext = torch.stack([
            nonneg_sin(t_ext, 0.0),
            nonneg_sin(t_ext, PHASE_OFFSET),
            nonneg_sin(t_ext, 2 * PHASE_OFFSET),
        ], dim=1).cpu().numpy()

    t_np = t_train.cpu().numpy()
    tgt_np = targets.cpu().numpy()
    t_ext_np = t_ext.cpu().numpy()

    # ---- Figure A: training fit + loss curve ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    for i in range(3):
        ax = axes[i // 2, i % 2] if i < 3 else None
    # redo layout cleanly
    plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(10, 9),
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.2]})
    for i in range(3):
        axes[i].plot(t_np, tgt_np[:, i], color=colors[i], lw=2, label="target")
        axes[i].plot(t_np, pred[:, i], color="k", lw=1.2, ls="--", label="GNN")
        axes[i].set_ylabel(f"node {i}")
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right", ncol=2)
    axes[2].set_xlabel("t")
    axes[3].semilogy(history, color="k", lw=1)
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("MSE loss")
    axes[3].set_title(f"Training loss — final = {history[-1]:.4f}")
    axes[3].grid(alpha=0.3, which="both")
    fig.suptitle(f"{name}: fit on training interval  [0, {T_MAX:.2f}]",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path_a = OUT_DIR / f"{name}_fit.png"
    fig.savefig(path_a, dpi=130)
    plt.close(fig)

    # ---- Figure B: extrapolation ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        axes[i].plot(t_ext_np, target_ext[:, i], color=colors[i], lw=2, label="target")
        axes[i].plot(t_ext_np, pred_ext[:, i], color="k", lw=1.2, ls="--", label="GNN")
        axes[i].axvspan(0, T_MAX, alpha=0.08, color="gray", label="train range")
        axes[i].set_ylabel(f"node {i}")
        axes[i].grid(alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right", ncol=3)
    axes[2].set_xlabel("t")
    subtitle = ("rollout beyond trained horizon" if not feedforward
                else "evaluated at unseen t values (no recurrence)")
    fig.suptitle(f"{name}: extrapolation — {subtitle}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path_b = OUT_DIR / f"{name}_extrapolation.png"
    fig.savefig(path_b, dpi=130)
    plt.close(fig)

    # MSE numbers
    mse_in = float(np.mean((pred - tgt_np) ** 2))
    mse_out = float(np.mean((pred_ext[N_STEPS:] - target_ext[N_STEPS:]) ** 2))
    print(f"  [{name}] MSE (in-range)    = {mse_in:.5f}")
    print(f"  [{name}] MSE (extrapolate) = {mse_out:.5f}")
    return path_a, path_b, mse_in, mse_out


# ═══════════════════════════════════════════════════════════════════════
# 7. Run both baselines
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Feedforward GNN — t as input, no temporal recurrence")
    print("=" * 60)
    ff = FeedforwardGNN(n_nodes=3, d_h=32, d_msg=32, n_layers=3)
    n_params_ff = sum(p.numel() for p in ff.parameters())
    print(f"  params: {n_params_ff}")
    hist_ff = train(ff, "feedforward_gnn", n_epochs=2000, lr=3e-3)
    fa, fb, mi, mo = plot_fit(ff, "feedforward_gnn", hist_ff, feedforward=True)

    print()
    print("=" * 60)
    print("Recurrent GNN — learned neural ODE on the 3-cycle graph")
    print("=" * 60)
    rc = RecurrentGNN(n_nodes=3, d_h=16, d_msg=32, n_layers=2)
    n_params_rc = sum(p.numel() for p in rc.parameters())
    print(f"  params: {n_params_rc}")
    hist_rc = train(rc, "recurrent_gnn", n_epochs=2000, lr=3e-3)
    ra, rb, mi2, mo2 = plot_fit(rc, "recurrent_gnn", hist_rc, feedforward=False)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  feedforward_gnn ({n_params_ff:>4} params):  in={mi:.4f}  extrap={mo:.4f}")
    print(f"  recurrent_gnn   ({n_params_rc:>4} params):  in={mi2:.4f}  extrap={mo2:.4f}")
    print()
    print("Outputs saved to", OUT_DIR)
