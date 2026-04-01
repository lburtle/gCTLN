import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

# ── Target: three signals 120° apart — achievable by a 3-cycle ───────────────
t_span       = torch.linspace(0, 6 * np.pi, 600)
phase_offset = 2 * np.pi / 3
targets = torch.stack([
    torch.sin(t_span),
    torch.sin(t_span + phase_offset),
    torch.sin(t_span + 2 * phase_offset)
], dim=1)

# ── Model ─────────────────────────────────────────────────────────────────────
class GCTLN(nn.Module):
    def __init__(self):
        super().__init__()
        # log-parameterize to keep positive
        # Initialize so delta > eps (limit cycle condition)
        self.log_eps   = nn.Parameter(torch.tensor(-2.0))  # eps ~ 0.13
        self.log_delta = nn.Parameter(torch.tensor(-0.5))  # delta ~ 0.60

        # Per-neuron tonic drive — inside the relu
        self.theta = nn.Parameter(torch.ones(3) * 0.1)

        # Linear readout: maps 3 neuron activities to 3 target signals
        # initialized to identity — can learn amplitude and sign
        self.readout = nn.Linear(3, 3, bias=True)
        nn.init.eye_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def get_eps_delta(self):
        eps   = torch.exp(self.log_eps)
        delta = torch.exp(self.log_delta)
        return eps, delta

    def build_W(self):
        eps, delta = self.get_eps_delta()
        # All entries start as non-edge: -1 - delta
        # Cyclic edges (j -> j+1 mod 3): -1 + eps
        W = torch.zeros(3, 3)
        for j in range(3):
            for i in range(3):
                if i == j:
                    W[i, j] = 0.0              # diagonal
                elif i == (j + 1) % 3:
                    W[i, j] = -1.0 + eps       # forward edge
                else:
                    W[i, j] = -1.0 - delta     # non-edge
        return W

    def ode_func(self, t, x):
        W     = self.build_W()
        drive = W @ x + self.theta
        return -x + torch.clamp(drive, min=0.0)

    def forward(self, t_eval, x0):
        traj = odeint(self.ode_func, x0, t_eval,
                      method='dopri5', rtol=1e-4, atol=1e-5)
        # traj: (T, 3) — apply linear readout
        pred = self.readout(traj)
        return traj, pred

    def limit_cycle_penalty(self):
        # Penalize eps >= delta (would give stable fixed point not limit cycle)
        eps, delta = self.get_eps_delta()
        violation  = torch.clamp(eps - delta + 0.02, min=0.0)
        return violation ** 2

    def print_params(self):
        eps, delta = self.get_eps_delta()
        print(f"  eps={eps.item():.4f}  delta={delta.item():.4f}  "
              f"delta>eps: {delta.item() > eps.item()}  "
              f"theta={self.theta.detach().numpy().round(3)}")


# ── Training ──────────────────────────────────────────────────────────────────
torch.manual_seed(42)
model     = GCTLN()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=600, gamma=0.3)

# Initial condition: slight asymmetry close to expected fixed point
x0 = torch.tensor([0.15, 0.08, 0.05])

n_epochs     = 2000
loss_history = []

for epoch in range(n_epochs):
    optimizer.zero_grad()

    traj, pred = model(t_span, x0)

    recon_loss = nn.functional.mse_loss(pred, targets)
    penalty    = model.limit_cycle_penalty()
    loss       = recon_loss + 20.0 * penalty

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    loss_history.append(recon_loss.item())

    if epoch % 200 == 0:
        print(f"epoch {epoch:4d} | loss {recon_loss.item():.5f}")
        model.print_params()

# ── Evaluation ────────────────────────────────────────────────────────────────
with torch.no_grad():
    traj, pred = model(t_span, x0)

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
phase_labels = ["sin(t)", "sin(t + 2π/3)", "sin(t + 4π/3)"]
for i, (ax, lbl) in enumerate(zip(axes, phase_labels)):
    ax.plot(t_span, targets[:, i],   "k--", lw=1.5, label="target")
    ax.plot(t_span, pred[:, i].numpy(), lw=1.5, label="gCTLN")
    ax.set_ylabel(lbl, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
axes[-1].set_xlabel("t")
plt.suptitle("gCTLN fitting 3 phase-shifted oscillations", fontsize=12)
plt.tight_layout()
plt.savefig("gctln_fit.png", dpi=150)

plt.figure(figsize=(8, 3))
plt.semilogy(loss_history)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Training loss")
plt.tight_layout()
plt.savefig("gctln_loss.png", dpi=150)

print("\nFinal parameters:")
model.print_params()
eps, delta = model.get_eps_delta()
print(f"Re(λ) = (delta - eps)/2 = "
      f"{(delta.item() - eps.item())/2:.4f}  "
      f"(positive = limit cycle)")

with torch.no_grad():
    W = model.build_W()
    eps, delta = model.get_eps_delta()
    print("Weight matrix W:")
    print(W.numpy().round(4))
    print(f"\neps={eps.item():.4f}  delta={delta.item():.4f}")
    print(f"Edge weight    (-1 + eps):   {(-1 + eps).item():.4f}")
    print(f"Non-edge weight (-1 - delta): {(-1 - delta).item():.4f}")