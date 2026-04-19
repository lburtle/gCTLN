# gctln_cmaes.py — Gradient-free training of a single gCTLN motif
#
# Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to
# optimize the 9 parameters of a 3-cycle gCTLN:
#   3 edge weights (eps_0, eps_1, eps_2)
#   3 non-edge weights (delta_0, delta_1, delta_2)
#   3 tonic drives (theta_0, theta_1, theta_2)
#
# CMA-ES is gradient-free: it evaluates the loss by running the ODE
# forward, no adjoint needed. This sidesteps gradient starvation,
# ReLU piecewise-linearity issues, and bifurcation traps entirely.

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from cmaes import CMA
import time


# ═══════════════════════════════════════════════════════════════════════
# 1. CTLN simulation
# ═══════════════════════════════════════════════════════════════════════

def build_W_3cycle(eps, delta):
    """
    Build weight matrix for a 3-cycle: 0->1->2->0

    Parameters
    ----------
    eps   : (3,) array — per-edge epsilon values
    delta : (3,) array — per-non-edge delta values

    Returns
    -------
    W : (3,3) weight matrix
    """
    # Adjacency: A[i,j]=True means j->i
    # 3-cycle: 0->1, 1->2, 2->0
    # So A[1,0]=True, A[2,1]=True, A[0,2]=True
    W = np.zeros((3, 3))

    # Edges (j->i): W[i,j] = -1 + eps
    W[1, 0] = -1.0 + eps[0]   # 0->1
    W[2, 1] = -1.0 + eps[1]   # 1->2
    W[0, 2] = -1.0 + eps[2]   # 2->0

    # Non-edges: W[i,j] = -1 - delta
    W[0, 1] = -1.0 - delta[0]  # 1-/>0
    W[1, 2] = -1.0 - delta[1]  # 2-/>1
    W[2, 0] = -1.0 - delta[2]  # 0-/>2

    # Diagonal = 0 (already)
    return W


def dxdt_ctln(x, t, W, theta):
    """CTLN dynamics: dx/dt = -x + [Wx + theta]_+"""
    return -x + np.maximum(W @ x + theta, 0)


def simulate(W, theta, T, n_steps, x0):
    """Run CTLN simulation, return (t, sol)."""
    t = np.linspace(0, T, n_steps)
    sol = odeint(dxdt_ctln, x0, t, args=(W, theta))
    return t, sol


# ═══════════════════════════════════════════════════════════════════════
# 2. Parameter encoding for CMA-ES
# ═══════════════════════════════════════════════════════════════════════

def decode_params(raw):
    """
    Map a 9-dimensional unconstrained vector to legal gCTLN parameters.

    raw[0:3] -> eps[0:3]   via sigmoid, mapped to (0.01, 0.99)
    raw[3:6] -> delta[0:3] via softplus, mapped to (0.01, inf)
    raw[6:9] -> theta[0:3] via softplus, mapped to (0.01, 1.0)

    Returns: eps (3,), delta (3,), theta (3,)
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def softplus(x):
        return np.log1p(np.exp(np.clip(x, -20, 20)))

    eps   = 0.01 + 0.98 * sigmoid(raw[0:3])     # in (0.01, 0.99)
    delta = 0.01 + softplus(raw[3:6])             # in (0.01, inf)
    theta = 0.01 + 0.99 * sigmoid(raw[6:9])       # in (0.01, 1.0)

    return eps, delta, theta


def encode_params(eps, delta, theta):
    """
    Inverse of decode_params. Maps known-good parameters to raw space
    for initialization.
    """
    def sigmoid_inv(y):
        y_clipped = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y_clipped / (1 - y_clipped))

    def softplus_inv(y):
        return np.log(np.expm1(np.clip(y, 1e-6, 50)))

    raw = np.zeros(9)
    raw[0:3] = sigmoid_inv((eps - 0.01) / 0.98)
    raw[3:6] = softplus_inv(delta - 0.01)
    raw[6:9] = sigmoid_inv((theta - 0.01) / 0.99)
    return raw


# ═══════════════════════════════════════════════════════════════════════
# 3. Loss function
# ═══════════════════════════════════════════════════════════════════════

def make_target(t_span):
    """
    Non-negative sinusoidal target with 120° phase offsets.
    Same as the training target we've been using.
    """
    phase_offset = 2 * np.pi / 3
    targets = np.stack([
        0.5 + 0.4 * np.sin(t_span),
        0.5 + 0.4 * np.sin(t_span + phase_offset),
        0.5 + 0.4 * np.sin(t_span + 2 * phase_offset),
    ], axis=1)
    return targets


def compute_loss(raw_params, t_span, targets, x0):
    """
    Evaluate the MSE loss for a given parameter vector.
    This runs the full ODE forward — no gradients needed.

    Returns a large penalty if the simulation fails (NaN, divergence).
    """
    try:
        eps, delta, theta = decode_params(raw_params)
        W = build_W_3cycle(eps, delta)
        t, sol = simulate(W, theta, T=t_span[-1], n_steps=len(t_span), x0=x0)

        # Check for simulation failure
        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)) or sol.max() > 100:
            return 1e6

        # MSE loss
        mse = np.mean((sol - targets) ** 2)

        # Bonus: penalize if activity dies (all neurons near-constant)
        late = sol[len(sol)//2:]
        variance = late.var(axis=0).mean()
        if variance < 1e-4:
            mse += 0.5  # nudge away from fixed points

        return mse

    except Exception:
        return 1e6


# ═══════════════════════════════════════════════════════════════════════
# 4. CMA-ES training loop
# ═══════════════════════════════════════════════════════════════════════

def train_cmaes(t_span, targets, x0,
                n_generations=500,
                population_size=30,
                sigma0=0.5,
                print_every=25):
    """
    Train gCTLN parameters using CMA-ES.

    Parameters
    ----------
    t_span     : (n_steps,) time points
    targets    : (n_steps, 3) target trajectories
    x0         : (3,) initial condition
    n_generations : max generations
    population_size : CMA-ES population
    sigma0     : initial step size
    print_every : logging frequency

    Returns
    -------
    best_params : raw parameter vector
    best_loss   : best loss achieved
    history     : list of (generation, best_loss) tuples
    """
    # Initialize from known-good CTLN parameters
    eps_init   = np.array([0.10, 0.10, 0.10])
    delta_init = np.array([0.50, 0.50, 0.50])
    theta_init = np.array([0.10, 0.10, 0.10])
    raw_init   = encode_params(eps_init, delta_init, theta_init)

    optimizer = CMA(
        mean=raw_init,
        sigma=sigma0,
        population_size=population_size,
    )

    best_loss = float('inf')
    best_params = raw_init.copy()
    history = []

    start_time = time.time()

    for gen in range(n_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            params = optimizer.ask()
            loss = compute_loss(params, t_span, targets, x0)
            solutions.append((params, loss))

        optimizer.tell(solutions)

        # Track best
        gen_best = min(solutions, key=lambda s: s[1])
        if gen_best[1] < best_loss:
            best_loss = gen_best[1]
            best_params = gen_best[0].copy()

        history.append((gen, best_loss))

        if gen % print_every == 0:
            eps, delta, theta = decode_params(best_params)
            elapsed = time.time() - start_time
            print(f"gen {gen:4d} | best_loss {best_loss:.6f} "
                  f"| eps [{eps[0]:.3f},{eps[1]:.3f},{eps[2]:.3f}] "
                  f"| theta [{theta[0]:.3f},{theta[1]:.3f},{theta[2]:.3f}] "
                  f"| {elapsed:.1f}s")

        # Early stopping
        if best_loss < 1e-4:
            print(f"Converged at generation {gen}!")
            break

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Best loss: {best_loss:.6f}")
    return best_params, best_loss, history


# ═══════════════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_results(best_params, t_span, targets, x0, history):
    """Plot the best solution against the target, plus training curve."""
    eps, delta, theta = decode_params(best_params)
    W = build_W_3cycle(eps, delta)
    t, sol = simulate(W, theta, T=t_span[-1], n_steps=len(t_span), x0=x0)

    # ── Fit plot ──────────────────────────────────────────────────────
    colors = ["#c0392b", "#1a5276", "#1e8449"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    for i, (ax, color) in enumerate(zip(axes, colors)):
        ax.plot(t_span, targets[:, i], "k--", lw=1.5, label="target")
        ax.plot(t_span, sol[:, i], color=color, lw=1.5, label="gCTLN")
        ax.set_ylabel(f"neuron {i}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("t")
    axes[0].set_title("gCTLN fit (CMA-ES)", fontsize=12)
    plt.tight_layout()
    plt.savefig("imgs/gctln_cmaes_fit.png", dpi=150)
    plt.close()

    # ── Training curve ────────────────────────────────────────────────
    gens, losses = zip(*history)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(gens, losses)
    ax.set_xlabel("generation")
    ax.set_ylabel("best MSE loss")
    ax.set_title("CMA-ES training progress")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("imgs/gctln_cmaes_loss.png", dpi=150)
    plt.close()

    # ── Print learned parameters ──────────────────────────────────────
    print(f"\nLearned parameters:")
    print(f"  eps   = [{eps[0]:.4f}, {eps[1]:.4f}, {eps[2]:.4f}]")
    print(f"  delta = [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}]")
    print(f"  theta = [{theta[0]:.4f}, {theta[1]:.4f}, {theta[2]:.4f}]")
    print(f"\nWeight matrix W:")
    print(np.round(W, 4))
    print(f"\nEdge weights (should be in (-1, 0)):")
    print(f"  0->1: {W[1,0]:.4f}")
    print(f"  1->2: {W[2,1]:.4f}")
    print(f"  2->0: {W[0,2]:.4f}")
    print(f"\nNon-edge weights (should be < -1):")
    print(f"  1-/>0: {W[0,1]:.4f}")
    print(f"  2-/>1: {W[1,2]:.4f}")
    print(f"  0-/>2: {W[2,0]:.4f}")

    # ── Verify oscillation in longer run ──────────────────────────────
    t_long, sol_long = simulate(W, theta, T=200, n_steps=6000, x0=x0)
    cut = len(t_long) // 2
    std_check = sol_long[cut:].std(axis=0)
    mean_check = sol_long[cut:].mean(axis=0)
    print(f"\nLong-run check (t=100..200):")
    print(f"  mean = {mean_check.round(4)}")
    print(f"  std  = {std_check.round(4)}")
    tag = "oscillating" if std_check.mean() > 0.01 else "static"
    print(f"  status: {tag}")

    return sol


# ═══════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Setup ─────────────────────────────────────────────────────────
    T = 10 * np.pi
    n_steps = 1000
    t_span = np.linspace(0, T, n_steps)
    targets = make_target(t_span)
    x0 = np.array([0.15, 0.08, 0.05])

    print(f"Target: non-negative sinusoids, T={T:.2f}, {n_steps} steps")
    print(f"Parameters to optimize: 9 (3 eps + 3 delta + 3 theta)")
    print(f"Method: CMA-ES (gradient-free)\n")

    # ── Verify initial parameters oscillate ───────────────────────────
    W_init = build_W_3cycle(
        eps=np.array([0.1, 0.1, 0.1]),
        delta=np.array([0.5, 0.5, 0.5]))
    t_check, sol_check = simulate(W_init, theta=np.array([0.1, 0.1, 0.1]),
                                   T=100, n_steps=3000, x0=x0)
    print(f"Initial params check (t=50..100):")
    cut = len(t_check) // 2
    print(f"  std = {sol_check[cut:].std(axis=0).round(4)}")
    print(f"  mean = {sol_check[cut:].mean(axis=0).round(4)}")
    init_loss = compute_loss(
        encode_params(np.array([0.1,0.1,0.1]),
                      np.array([0.5,0.5,0.5]),
                      np.array([0.1,0.1,0.1])),
        t_span, targets, x0)
    print(f"  initial loss: {init_loss:.6f}\n")

    # ── Train ─────────────────────────────────────────────────────────
    best_params, best_loss, history = train_cmaes(
        t_span, targets, x0,
        n_generations=500,
        population_size=30,
        sigma0=0.5,
        print_every=25,
    )

    # ── Visualize ─────────────────────────────────────────────────────
    sol = plot_results(best_params, t_span, targets, x0, history)
    print("\nSaved: gctln_cmaes_fit.png, gctln_cmaes_loss.png")