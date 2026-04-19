# Two-stage gCTLN training
#
# Stage 1 (CMA-ES): Find CTLN parameters that produce a stable limit
#   cycle with the right frequency. Gradient-free, no adjoint issues.
#   Optimizes 9 params: 3 eps, 3 delta, 3 theta.
#   Loss = frequency match + oscillation amplitude reward.
#
# Stage 2 (Gradient descent): Freeze the CTLN, train a linear readout
#   to map the raw piecewise-linear CTLN waveform to the smooth
#   sinusoidal target. This is a simple least-squares problem.
#
# The insight: CTLNs provide oscillatory structure (frequency, phase,
# sequential activation). The readout provides waveform shaping
# (amplitude, offset, smoothing). Separating these two concerns
# makes both problems tractable.

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from cmaes import CMA
import time


# ═══════════════════════════════════════════════════════════════════════
# 1. CTLN simulation (numpy, for CMA-ES)
# ═══════════════════════════════════════════════════════════════════════

def build_W_3cycle(eps, delta):
    """Build weight matrix for 3-cycle: 0->1->2->0"""
    W = np.zeros((3, 3))
    W[1, 0] = -1.0 + eps[0]   # 0->1
    W[2, 1] = -1.0 + eps[1]   # 1->2
    W[0, 2] = -1.0 + eps[2]   # 2->0
    W[0, 1] = -1.0 - delta[0]  # 1-/>0
    W[1, 2] = -1.0 - delta[1]  # 2-/>1
    W[2, 0] = -1.0 - delta[2]  # 0-/>2
    return W


def dxdt_ctln(x, t, W, theta):
    """CTLN dynamics: dx/dt = -x + [Wx + theta]_+"""
    return -x + np.maximum(W @ x + theta, 0)


def simulate(W, theta, T, n_steps, x0):
    t = np.linspace(0, T, n_steps)
    sol = odeint(dxdt_ctln, x0, t, args=(W, theta))
    return t, sol


# ═══════════════════════════════════════════════════════════════════════
# 2. Parameter encoding
# ═══════════════════════════════════════════════════════════════════════

def decode_params(raw):
    """Map unconstrained 9-vector to legal gCTLN parameters."""
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    def softplus(x):
        return np.log1p(np.exp(np.clip(x, -20, 20)))

    eps   = 0.01 + 0.98 * sigmoid(raw[0:3])
    delta = 0.01 + softplus(raw[3:6])
    theta = 0.01 + 0.99 * sigmoid(raw[6:9])
    return eps, delta, theta


def encode_params(eps, delta, theta):
    """Inverse of decode_params."""
    def sigmoid_inv(y):
        y = np.clip(y, 1e-6, 1 - 1e-6)
        return np.log(y / (1 - y))
    def softplus_inv(y):
        return np.log(np.expm1(np.clip(y, 1e-6, 50)))

    raw = np.zeros(9)
    raw[0:3] = sigmoid_inv((eps - 0.01) / 0.98)
    raw[3:6] = softplus_inv(delta - 0.01)
    raw[6:9] = sigmoid_inv((theta - 0.01) / 0.99)
    return raw


# ═══════════════════════════════════════════════════════════════════════
# 3. Stage 1: CMA-ES for oscillation structure
# ═══════════════════════════════════════════════════════════════════════

def estimate_frequency(signal, dt):
    """Estimate oscillation frequency from peak spacing."""
    signal = signal - signal.mean()
    peaks, _ = find_peaks(signal, height=0.01 * signal.std(),
                          distance=int(0.5 / dt))
    if len(peaks) < 2:
        return 0.0
    periods = np.diff(peaks) * dt
    return 1.0 / periods.mean()


def stage1_loss(raw_params, target_freq, T, n_steps, x0):
    """
    CMA-ES loss for Stage 1.
    
    We want:
      1. Stable oscillation (high variance in late trajectory)
      2. Correct frequency (matching target sinusoid)
      3. All three neurons participating (no winner-take-all)
    
    We do NOT care about waveform shape — the readout handles that.
    """
    try:
        eps, delta, theta = decode_params(raw_params)
        W = build_W_3cycle(eps, delta)
        t, sol = simulate(W, theta, T=T, n_steps=n_steps, x0=x0)

        if np.any(np.isnan(sol)) or np.any(np.isinf(sol)) or sol.max() > 100:
            return 1e6

        # Use second half (after transient)
        half = len(t) // 2
        dt = t[1] - t[0]
        late = sol[half:]

        # 1. Variance reward — all neurons should oscillate
        per_neuron_var = late.var(axis=0)
        min_var = per_neuron_var.min()
        mean_var = per_neuron_var.mean()

        if mean_var < 1e-4:
            return 1e6  # dead network, maximum penalty

        # Penalize if any neuron is much less active than others
        var_balance = min_var / (mean_var + 1e-8)

        # 2. Frequency match — use neuron 0 as reference
        freq = estimate_frequency(late[:, 0], dt)
        if freq == 0:
            return 1e6  # no oscillation detected

        freq_error = (freq - target_freq) ** 2

        # 3. Phase offset check — neurons should fire sequentially
        # In a 3-cycle, peaks of neuron i should lead neuron i+1 by ~1/3 period
        peaks_0, _ = find_peaks(late[:, 0] - late[:, 0].mean(),
                                distance=int(0.3 / dt))
        peaks_1, _ = find_peaks(late[:, 1] - late[:, 1].mean(),
                                distance=int(0.3 / dt))
        if len(peaks_0) >= 2 and len(peaks_1) >= 2:
            # Average phase difference
            n_compare = min(len(peaks_0), len(peaks_1))
            phase_diffs = (peaks_1[:n_compare] - peaks_0[:n_compare]) * dt
            mean_period = 1.0 / freq
            # We want phase diff ≈ period/3
            phase_error = (np.mean(np.abs(phase_diffs)) - mean_period / 3) ** 2
        else:
            phase_error = 0.1  # can't measure, mild penalty

        # Combined loss
        loss = (10.0 * freq_error           # frequency is most important
                + 1.0 * phase_error          # phase relationships
                - 0.5 * mean_var             # reward variance (negative = good)
                - 0.3 * var_balance)         # reward balanced activation

        return loss

    except Exception:
        return 1e6


def train_stage1(target_freq, T, n_steps, x0,
                 n_generations=300,
                 population_size=40,
                 sigma0=0.5,
                 print_every=25):
    """
    Stage 1: Find CTLN parameters that oscillate at the target frequency
    with balanced, sequential activation.
    """
    print("=" * 60)
    print("STAGE 1: CMA-ES for oscillation structure")
    print(f"  Target frequency: {target_freq:.4f} Hz")
    print(f"  Population: {population_size}, Generations: {n_generations}")
    print("=" * 60)

    # Initialize from standard CTLN parameters
    raw_init = encode_params(
        np.array([0.10, 0.10, 0.10]),
        np.array([0.50, 0.50, 0.50]),
        np.array([0.10, 0.10, 0.10]))

    optimizer = CMA(mean=raw_init, sigma=sigma0,
                    population_size=population_size)

    best_loss = float('inf')
    best_params = raw_init.copy()
    history = []
    start = time.time()

    for gen in range(n_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            params = optimizer.ask()
            loss = stage1_loss(params, target_freq, T, n_steps, x0)
            solutions.append((params, loss))

        optimizer.tell(solutions)

        gen_best = min(solutions, key=lambda s: s[1])
        if gen_best[1] < best_loss:
            best_loss = gen_best[1]
            best_params = gen_best[0].copy()

        history.append((gen, best_loss))

        if gen % print_every == 0:
            eps, delta, theta = decode_params(best_params)
            freq = estimate_freq_from_params(best_params, T, n_steps, x0)
            print(f"  gen {gen:4d} | loss {best_loss:+.4f} "
                  f"| freq {freq:.4f} "
                  f"| eps [{eps[0]:.3f},{eps[1]:.3f},{eps[2]:.3f}] "
                  f"| delta [{delta[0]:.2f},{delta[1]:.2f},{delta[2]:.2f}] "
                  f"| theta [{theta[0]:.3f},{theta[1]:.3f},{theta[2]:.3f}] "
                  f"| {time.time()-start:.1f}s")

    print(f"\n  Stage 1 done in {time.time()-start:.1f}s. Best loss: {best_loss:.4f}")
    return best_params, history


def estimate_freq_from_params(raw_params, T, n_steps, x0):
    """Helper: simulate and estimate frequency."""
    eps, delta, theta = decode_params(raw_params)
    W = build_W_3cycle(eps, delta)
    t, sol = simulate(W, theta, T=T, n_steps=n_steps, x0=x0)
    dt = t[1] - t[0]
    half = len(t) // 2
    return estimate_frequency(sol[half:, 0], dt)


# ═══════════════════════════════════════════════════════════════════════
# 4. Stage 2: Gradient descent on readout layer
# ═══════════════════════════════════════════════════════════════════════

def train_stage2(ctln_params, t_span, targets, x0,
                 n_epochs=2000, lr=1e-2, print_every=200):
    """
    Stage 2: Freeze CTLN parameters, train a linear readout to map
    the raw CTLN waveform to the target.
    
    This is essentially a least-squares regression:
      pred(t) = readout_weight @ ctln_traj(t) + readout_bias
    
    But we train it with gradient descent so it generalizes
    to deeper readouts if needed later.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Gradient descent on readout layer")
    print("=" * 60)

    # Generate the fixed CTLN trajectory
    eps, delta, theta = decode_params(ctln_params)
    W = build_W_3cycle(eps, delta)
    t, sol = simulate(W, theta, T=t_span[-1], n_steps=len(t_span), x0=x0)
    ctln_traj = torch.from_numpy(sol).float()  # (n_steps, 3)
    target_tensor = torch.from_numpy(targets).float()

    # Linear readout: 3 -> 3 (with bias)
    readout = nn.Linear(3, 3)
    nn.init.eye_(readout.weight)
    nn.init.zeros_(readout.bias)

    optimizer = torch.optim.Adam(readout.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.3)

    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = readout(ctln_traj)
        loss = nn.functional.mse_loss(pred, target_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss.item())

        if epoch % print_every == 0:
            print(f"  epoch {epoch:4d} | MSE {loss.item():.6f}")

    print(f"\n  Stage 2 done. Final MSE: {loss_history[-1]:.6f}")

    # Get final prediction
    with torch.no_grad():
        final_pred = readout(ctln_traj).numpy()

    # Print readout parameters
    with torch.no_grad():
        print(f"\n  Readout weight:\n{readout.weight.numpy().round(4)}")
        print(f"  Readout bias: {readout.bias.numpy().round(4)}")

    return readout, final_pred, ctln_traj.numpy(), loss_history


# ═══════════════════════════════════════════════════════════════════════
# 5. Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_results(t_span, targets, ctln_traj, final_pred,
                 stage1_history, stage2_history):
    """Plot everything: raw CTLN, readout output, targets, training curves."""

    colors = ["#c0392b", "#1a5276", "#1e8449"]

    # ── Main fit plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for i, (ax, color) in enumerate(zip(axes, colors)):
        ax.plot(t_span, targets[:, i], "k--", lw=1.5, label="target")
        ax.plot(t_span, ctln_traj[:, i], color=color, lw=0.8,
                alpha=0.4, linestyle=":", label="raw CTLN")
        ax.plot(t_span, final_pred[:, i], color=color, lw=1.5,
                label="readout output")
        ax.set_ylabel(f"neuron {i}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("t")
    axes[0].set_title(
        "gCTLN two-stage training: CMA-ES (structure) + readout (shaping)",
        fontsize=12)
    plt.tight_layout()
    plt.savefig("imgs/gctln_twostage_fit.png", dpi=150)
    plt.close()

    # ── Training curves ───────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))

    gens, losses = zip(*stage1_history)
    ax1.plot(gens, losses)
    ax1.set_xlabel("generation")
    ax1.set_ylabel("stage 1 loss")
    ax1.set_title("Stage 1: CMA-ES (oscillation structure)")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.semilogy(stage2_history)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("MSE")
    ax2.set_title("Stage 2: readout training (waveform shaping)")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("imgs/gctln_twostage_loss.png", dpi=150)
    plt.close()

    # ── Raw CTLN waveform detail ──────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    # Show just the last 40% to see the limit cycle clearly
    cut = int(0.6 * len(t_span))
    t_late = t_span[cut:]
    sol_late = ctln_traj[cut:]

    for i, (ax, color) in enumerate(zip(axes, colors)):
        ax.plot(t_late, sol_late[:, i], color=color, lw=1.2)
        ax.set_ylabel(f"neuron {i}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("t")
    axes[0].set_title("Raw CTLN firing rates (late trajectory)", fontsize=12)
    plt.tight_layout()
    plt.savefig("imgs/gctln_twostage_raw.png", dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Setup ─────────────────────────────────────────────────────────
    T = 10 * np.pi
    n_steps = 1000
    t_span = np.linspace(0, T, n_steps)
    x0 = np.array([0.15, 0.08, 0.05])

    # Target: non-negative sinusoids with 120-degree phase offsets
    phase_offset = 2 * np.pi / 3
    targets = np.stack([
        0.5 + 0.4 * np.sin(t_span),
        0.5 + 0.4 * np.sin(t_span + phase_offset),
        0.5 + 0.4 * np.sin(t_span + 2 * phase_offset),
    ], axis=1)

    # Target frequency: sin has period 2*pi, so freq = 1/(2*pi)
    target_freq = 1.0 / (2 * np.pi)
    dt = t_span[1] - t_span[0]

    print(f"Target: non-negative sinusoids")
    print(f"  Period: {2*np.pi:.2f},  Frequency: {target_freq:.4f} Hz")
    print(f"  Amplitude: 0.4,  Offset: 0.5")
    print(f"  Time span: [0, {T:.2f}],  Steps: {n_steps}")
    print(f"  Phase offsets: 0, {phase_offset:.2f}, {2*phase_offset:.2f} rad")
    print()

    # ── Stage 1: CMA-ES for oscillation ───────────────────────────────
    # Use a longer simulation for frequency estimation stability
    T_stage1 = 30 * np.pi  # ~15 cycles for reliable freq estimate
    n_steps_s1 = 3000

    best_params, s1_history = train_stage1(
        target_freq=target_freq,
        T=T_stage1,
        n_steps=n_steps_s1,
        x0=x0,
        n_generations=400,
        population_size=50,
        sigma0=0.5,
        print_every=25,
    )

    # Print what Stage 1 found
    eps, delta, theta = decode_params(best_params)
    W = build_W_3cycle(eps, delta)
    print(f"\n  Stage 1 result:")
    print(f"    eps   = {eps.round(4)}")
    print(f"    delta = {delta.round(4)}")
    print(f"    theta = {theta.round(4)}")
    print(f"    W =\n{np.round(W, 4)}")

    # Verify oscillation
    t_check, sol_check = simulate(W, theta, T=200, n_steps=6000, x0=x0)
    cut = len(t_check) // 2
    std_check = sol_check[cut:].std(axis=0)
    freq_check = estimate_frequency(sol_check[cut:, 0],
                                     t_check[1] - t_check[0])
    print(f"    Oscillation std: {std_check.round(4)}")
    print(f"    Measured freq: {freq_check:.4f} (target: {target_freq:.4f})")

    # ── Stage 2: Train readout ────────────────────────────────────────
    readout, final_pred, ctln_traj, s2_history = train_stage2(
        best_params, t_span, targets, x0,
        n_epochs=3000, lr=1e-2, print_every=300)

    # ── Visualize ─────────────────────────────────────────────────────
    plot_results(t_span, targets, ctln_traj, final_pred,
                 s1_history, s2_history)

    # ── Final summary ─────────────────────────────────────────────────
    mse = np.mean((final_pred - targets) ** 2)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"  Overall MSE: {mse:.6f}")
    print(f"  CTLN params: 9 (frozen after stage 1)")
    print(f"  Readout params: {sum(p.numel() for p in readout.parameters())}")
    print(f"  Saved: gctln_twostage_fit.png")
    print(f"         gctln_twostage_loss.png")
    print(f"         gctln_twostage_raw.png")