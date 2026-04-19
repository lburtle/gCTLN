# Imports

import numpy as np
from scipy.integrate import odeint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Dynamics correctly

def dxdt_gctln(x, t, W, theta):
    """
    CTLN dynamics: dx/dt = -x + relu(W @ x + theta)
    theta is ADDED INSIDE the relu — this is Curto's standard formulation.
    Ensures neurons have tonic drive that inhibition must overcome.
    """
    return -x + np.maximum(W @ x + theta, 0)

# Network construction

def cyclic_tournament(n):
    A = np.zeros((n, n), dtype=bool)
    for j in range(n):
        A[(j + 1) % n, j] = True
    return A

def motif_block(n, eps, delta):
    A = cyclic_tournament(n)
    W = np.where(A, -1.0 + eps, -1.0 - delta).astype(float)
    np.fill_diagonal(W, 0.0)
    return W

def cross_block(n_target, n_source, delta_cross):
    return np.full((n_target, n_source), -1.0 - delta_cross)

def build_gctln(motif_sizes, eps, delta, delta_cross):
    N      = sum(motif_sizes)
    W      = np.zeros((N, N))
    starts = np.concatenate(([0], np.cumsum(motif_sizes[:-1]))).astype(int)
    slices = [slice(s, s + sz) for s, sz in zip(starts, motif_sizes)]
    for i, (si, ni) in enumerate(zip(slices, motif_sizes)):
        for j, (sj, nj) in enumerate(zip(slices, motif_sizes)):
            if i == j:
                W[si, si] = motif_block(ni, eps, delta)
            else:
                W[si, sj] = cross_block(ni, nj, delta_cross)
    return W, slices

# Diagnostics

def diagnose_motif(n, eps, delta, theta):
    A = np.zeros((n, n), dtype=bool)
    for j in range(n):
        A[(j+1) % n, j] = True
    W = np.where(A, -1.0 + eps, -1.0 - delta).astype(float)
    np.fill_diagonal(W, 0.0)
    A_mat   = W - np.eye(n)
    eigvals = np.linalg.eigvals(A_mat)
    try:
        x_star = np.linalg.solve(np.eye(n) - W, theta * np.ones(n))
    except np.linalg.LinAlgError:
        x_star = None
    print(f"\n{'='*50}")
    print(f"  {n}-cycle | eps={eps}, delta={delta}, theta={theta}")
    print(f"{'='*50}")
    for ev in sorted(eigvals, key=lambda x: -x.real):
        tag = "UNSTABLE" if ev.real > 0 else "stable"
        osc = "oscillatory" if abs(ev.imag) > 1e-10 else "real"
        print(f"  {ev.real:+.4f} {ev.imag:+.4f}i  →  {tag}, {osc}")
    if x_star is not None:
        print(f"  Fixed point x*: {np.round(x_star, 4)}")
    max_real   = max(ev.real for ev in eigvals)
    has_complex = any(abs(ev.imag) > 1e-10 for ev in eigvals)
    prediction = (
        "LIMIT CYCLE ✓"    if max_real > 0 and has_complex
        else "STABLE FIXED POINT ✗"
    )
    print(f"  Prediction: {prediction}")
    print(f"  Re(λ) check: δ - ε / 2 = {(delta - eps)/2:+.4f}")
    return eigvals, x_star

# Simulation and plotting

def simulate(W, theta, T, n_steps, x0):
    t   = np.linspace(0, T, n_steps)
    sol = odeint(dxdt_gctln, x0, t, args=(W, theta))
    return t, sol

def diagnose_and_plot(W, slices, x0, theta=0.1, T=500, n_steps=10000,
                      motif_labels=None, skip_transient_frac=0.5):
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(len(slices))]
    t, sol = simulate(W, theta, T, n_steps, x0)
    cut    = int(n_steps * skip_transient_frac)
    t_ss   = t[cut:]
    sol_ss = sol[cut:]

    print(f"Plotting t = {t_ss[0]:.1f} to {t_ss[-1]:.1f} "
          f"({len(t_ss)} steps after transient removed)")
    print(f"Firing rate range in steady state: "
          f"[{sol_ss.min():.4f}, {sol_ss.max():.4f}]")
    print(f"Firing rate std per neuron: {sol_ss.std(axis=0).round(4)}")

    for idx, sl in enumerate(slices):
        mean_std = sol_ss[:, sl].std(axis=0).mean()
        tag = "oscillating" if mean_std > 0.01 else "static (fixed point)"
        print(f"  {motif_labels[idx]} mean firing rate std: "
              f"{mean_std:.4f} ← {tag}")

    # time series
    palettes = [
        ["#c0392b","#e74c3c","#f1948a"],
        ["#1a5276","#2471a3","#7fb3d3"],
        ["#1e8449","#27ae60","#82e0aa"],
    ]
    k    = len(slices)
    fig, axes = plt.subplots(k, 1, figsize=(14, 3*k), sharex=True)
    if k == 1:
        axes = [axes]
    for idx, (sl, ax) in enumerate(zip(slices, axes)):
        pal      = palettes[idx % len(palettes)]
        n_neurons = sl.stop - sl.start
        for ni, color in zip(range(sl.start, sl.stop), pal[:n_neurons]):
            ax.plot(t_ss, sol_ss[:, ni], color=color,
                    lw=1.2, label=f"n{ni - sl.start}")
        ax.set_ylabel("Firing rate", fontsize=10)
        ax.set_title(motif_labels[idx], fontsize=11, loc="left")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.4)
        ax.set_ylim(bottom=0)
        ax.spines[["top","right"]].set_visible(False)
    axes[-1].set_xlabel("Time (steady state)", fontsize=10)
    plt.tight_layout()
    plt.close("all")

    # PCA
    pca     = PCA(n_components=3)
    sol_pca = pca.fit_transform(sol_ss)
    var     = pca.explained_variance_ratio_ * 100
    print(f"\nPCA variance explained: "
          f"PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, PC3={var[2]:.1f}%")
    print(f"  (if PC1+PC2 ≈ 100%, the cycle lives in a 2D plane)")

    fig2 = plt.figure(figsize=(9, 7))
    ax3d = fig2.add_subplot(111, projection="3d")
    colors_time = plt.cm.viridis(np.linspace(0, 1, len(sol_pca)))
    for i in range(len(sol_pca) - 1):
        ax3d.plot(sol_pca[i:i+2, 0], sol_pca[i:i+2, 1], sol_pca[i:i+2, 2],
                  color=colors_time[i], lw=0.8, alpha=0.7)
    ax3d.scatter(*sol_pca[0],  color="green", s=60, zorder=5, label="start")
    ax3d.scatter(*sol_pca[-1], color="red",   s=60, zorder=5, label="end")
    ax3d.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=9)
    ax3d.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=9)
    ax3d.set_zlabel(f"PC3 ({var[2]:.1f}%)", fontsize=9)
    ax3d.legend(fontsize=9)
    plt.tight_layout()
    plt.close("all")

    return t, sol, sol_ss, t_ss, sol_pca

# eps   = 0.10   # or whatever produced your limit cycle
# delta = 0.50
# theta = 0.10   # or your working theta

# # Verify analytically first
# diagnose_motif(n=3, eps=eps, delta=delta, theta=theta)

# # ── Sweep delta_cross to find the three regimes ───────────────────────────
# for dc in [0.10, 0.30, 0.50, 0.70]:
#     W, slices = build_gctln([3, 3], eps=eps, delta=delta, delta_cross=dc)

#     # Symmetric x0 — neither motif favoured
#     x0 = np.array([0.3, 0.2, 0.1,
#                    0.3, 0.2, 0.1])

#     print(f"\n{'='*60}")
#     print(f"delta_cross = {dc}")
#     t, sol, sol_ss, t_ss, sol_pca = diagnose_and_plot(
#         W, slices, x0,
#         theta              = theta,
#         T                  = 500,
#         n_steps            = 15000,
#         motif_labels       = ["CPG-A", "CPG-B"],
#         skip_transient_frac = 0.6
#     )

# eps   = 0.10
# delta = 0.50
# theta = 0.10

# # Favour A strongly — should always win regardless of dc
# x0_favour_A = np.array([0.5, 0.1, 0.05,
#                          0.05, 0.05, 0.05])

# # Favour B strongly
# x0_favour_B = np.array([0.05, 0.05, 0.05,
#                          0.5,  0.1,  0.05])

# # Symmetric — let competition decide
# x0_symmetric = np.array([0.3, 0.2, 0.1,
#                           0.3, 0.2, 0.1])

# for dc in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
#     print(f"\n{'#'*60}")
#     print(f"delta_cross = {dc}")
#     W, slices = build_gctln([3, 3], eps=eps, delta=delta,
#                              delta_cross=dc)

#     for label, x0 in [("Favour A", x0_favour_A),
#                        ("Favour B", x0_favour_B),
#                        ("Symmetric", x0_symmetric)]:
#         t, sol = simulate(W, theta, T=800, n_steps=20000, x0=x0)
#         cut    = int(0.7 * len(t))
#         sol_ss = sol[cut:]

#         std_A = sol_ss[:, slices[0]].std(axis=0).mean()
#         std_B = sol_ss[:, slices[1]].std(axis=0).mean()
#         mean_A = sol_ss[:, slices[0]].mean()
#         mean_B = sol_ss[:, slices[1]].mean()

#         winner = ("A wins" if mean_A > mean_B * 2
#                   else "B wins" if mean_B > mean_A * 2
#                   else "coexisting")

#         print(f"  {label:12s} | "
#               f"std_A={std_A:.4f} std_B={std_B:.4f} | "
#               f"mean_A={mean_A:.4f} mean_B={mean_B:.4f} | "
#               f"{winner}")

# W, slices = build_gctln([3, 3], eps=eps, delta=delta, delta_cross=0.05)

# t, sol, sol_ss, t_ss, sol_pca = diagnose_and_plot(
#     W, slices,
#     x0                  = x0_symmetric,
#     theta               = theta,
#     T                   = 800,        # long enough to see who wins
#     n_steps             = 40000,
#     motif_labels        = ["CPG-A", "CPG-B"],
#     skip_transient_frac = 0.3         # keep more — the competition IS the story
# )

# Find delta cross bifurcation (transitions)

def bifurcation_sweep(eps, delta, theta, n_dc=200, T=600, n_steps=18000,
                      transient_frac=0.6):
    """
    Sweep delta_cross finely. For each value run TWO simulations:
    one favouring A, one favouring B.
    
    Record mean activity of each motif in steady state.
    
    Transition 1 (coexisting -> multistable):
        A-favour and B-favour give DIFFERENT outcomes
    Transition 2 (multistable -> WTA):
        The losing motif goes fully silent (mean ~ 0)
    """
    dc_vals = np.linspace(0.001, 0.30, n_dc)

    mean_A_fromA = np.zeros(n_dc)  # mean of A when started favouring A
    mean_B_fromA = np.zeros(n_dc)  # mean of B when started favouring A
    mean_A_fromB = np.zeros(n_dc)  # mean of A when started favouring B
    mean_B_fromB = np.zeros(n_dc)  # mean of B when started favouring B

    x0_A = np.array([0.6, 0.2, 0.1, 0.05, 0.05, 0.05])
    x0_B = np.array([0.05, 0.05, 0.05, 0.6, 0.2, 0.1])

    cut = int(n_steps * transient_frac)

    for k, dc in enumerate(dc_vals):
        W, slices = build_gctln([3, 3], eps=eps, delta=delta,
                                 delta_cross=dc)

        # Run from A-favoured start
        t, sol = simulate(W, theta, T, n_steps, x0_A)
        ss = sol[cut:]
        mean_A_fromA[k] = ss[:, slices[0]].mean()
        mean_B_fromA[k] = ss[:, slices[1]].mean()

        # Run from B-favoured start
        t, sol = simulate(W, theta, T, n_steps, x0_B)
        ss = sol[cut:]
        mean_A_fromB[k] = ss[:, slices[0]].mean()
        mean_B_fromB[k] = ss[:, slices[1]].mean()

    # ── Detect transitions ────────────────────────────────────────────────
    # Difference between the two runs — nonzero means initial conditions matter
    diff_A = np.abs(mean_A_fromA - mean_A_fromB)
    diff_B = np.abs(mean_B_fromA - mean_B_fromB)
    sensitivity = (diff_A + diff_B) / 2

    # Transition 1: sensitivity rises from zero
    threshold_1 = 0.001
    above_thresh = sensitivity > threshold_1
    if above_thresh.any():
        t1_idx = np.where(above_thresh)[0][0]
        t1     = dc_vals[t1_idx]
        print(f"Transition 1 (coexisting → multistable): "
              f"delta_cross ≈ {t1:.4f}")
    else:
        t1 = None
        print("Transition 1 not found in range")

    # Transition 2: losing motif goes silent (mean < 0.001)
    loser_silent = (mean_B_fromA < 0.001) & (mean_A_fromB < 0.001)
    if loser_silent.any():
        t2_idx = np.where(loser_silent)[0][0]
        t2     = dc_vals[t2_idx]
        print(f"Transition 2 (multistable → winner-take-all): "
              f"delta_cross ≈ {t2:.4f}")
    else:
        t2 = None
        print("Transition 2 not found in range — widen dc range")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    # Panel 1: mean activity from A-favoured start
    axes[0].plot(dc_vals, mean_A_fromA, color="#c0392b", lw=1.5,
                 label="CPG-A")
    axes[0].plot(dc_vals, mean_B_fromA, color="#1a5276", lw=1.5,
                 label="CPG-B")
    axes[0].set_ylabel("Mean firing rate", fontsize=10)
    axes[0].set_title("Started favouring A", fontsize=10, loc="left")
    axes[0].legend(fontsize=9)

    # Panel 2: mean activity from B-favoured start
    axes[1].plot(dc_vals, mean_A_fromB, color="#c0392b", lw=1.5,
                 label="CPG-A")
    axes[1].plot(dc_vals, mean_B_fromB, color="#1a5276", lw=1.5,
                 label="CPG-B")
    axes[1].set_ylabel("Mean firing rate", fontsize=10)
    axes[1].set_title("Started favouring B", fontsize=10, loc="left")
    axes[1].legend(fontsize=9)

    # Panel 3: sensitivity to initial conditions
    axes[2].plot(dc_vals, sensitivity, color="#1e8449", lw=1.5,
                 label="Sensitivity to x0")
    axes[2].set_ylabel("Mean |difference|", fontsize=10)
    axes[2].set_xlabel("delta_cross", fontsize=11)
    axes[2].set_title("Initial condition sensitivity "
                      "(nonzero = multistable)", fontsize=10, loc="left")

    # Mark transitions
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        if t1 is not None:
            ax.axvline(t1, color="#e67e22", lw=1.5, linestyle="--",
                       label=f"T1≈{t1:.3f}")
        if t2 is not None:
            ax.axvline(t2, color="#8e44ad", lw=1.5, linestyle=":",
                       label=f"T2≈{t2:.3f}")

    fig.suptitle(
        f"Bifurcation diagram  |  eps={eps}, delta={delta}, theta={theta}",
        fontsize=12)
    plt.tight_layout()
    plt.close("all")

    return dc_vals, mean_A_fromA, mean_B_fromA, mean_A_fromB, mean_B_fromB


# dc_vals, mAA, mBA, mAB, mBB = bifurcation_sweep(
#     eps=0.10, delta=0.50, theta=0.10
# )

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from scipy.integrate import odeint
from itertools import combinations

# ── Scalable color palette ────────────────────────────────────────────────────

def make_palette(k):
    """
    Generate k visually distinct colors.
    For k <= 6 uses hand-picked colors, beyond that samples a colormap.
    """
    base = ["#c0392b", "#1a5276", "#1e8449",
            "#7d3c98", "#d35400", "#117a65"]
    if k <= len(base):
        return base[:k]
    cmap   = plt.cm.tab20
    return [mcolors.to_hex(cmap(i / k)) for i in range(k)]

def make_neuron_palette(slices, motif_colors):
    """
    For each neuron, return a color that is a shade of its motif's color.
    Neurons within the same motif get progressively lighter shades.
    """
    colors = []
    for idx, sl in enumerate(slices):
        n      = sl.stop - sl.start
        base   = np.array(mcolors.to_rgb(motif_colors[idx]))
        for ni in range(n):
            # shade from full color (first neuron) to 60% lighter (last)
            t = ni / max(n - 1, 1)
            shade = base + t * (np.ones(3) - base) * 0.6
            colors.append(mcolors.to_hex(np.clip(shade, 0, 1)))
    return colors

# ── Per-subgraph PCA ──────────────────────────────────────────────────────────

def plot_per_subgraph_pca(sol_ss, slices, motif_labels=None,
                          color_by_motif=None, ncols=4):
    """
    One 3D PCA panel per motif. Panels wrap into rows of ncols.
    """
    k = len(slices)
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(k)]
    motif_colors = make_palette(k)

    ncols = min(ncols, k)
    nrows = int(np.ceil(k / ncols))
    fig   = plt.figure(figsize=(5 * ncols, 4.5 * nrows))

    for idx, sl in enumerate(slices):
        ax  = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        sub = sol_ss[:, sl]
        n_components = min(3, sub.shape[1])
        pca  = PCA(n_components=n_components)
        proj = pca.fit_transform(sub)
        var  = pca.explained_variance_ratio_ * 100

        if (color_by_motif is not None
                and color_by_motif != idx
                and color_by_motif < k):
            sig = sol_ss[:, slices[color_by_motif]].mean(axis=1)
            sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-10)
            colors = plt.cm.coolwarm(sig)
            clabel = f"by {motif_labels[color_by_motif]}"
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(proj)))
            clabel = "by time"

        pad = np.zeros(len(proj))
        c1  = proj[:, 0]
        c2  = proj[:, 1]
        c3  = proj[:, 2] if n_components == 3 else pad

        for i in range(len(proj) - 1):
            ax.plot(c1[i:i+2], c2[i:i+2], c3[i:i+2],
                    color=colors[i], lw=0.8, alpha=0.7)

        ax.set_title(f"{motif_labels[idx]}\n({clabel})",
                     fontsize=9, color=motif_colors[idx])
        ax.set_xlabel(f"PC1 {var[0]:.0f}%", fontsize=7, labelpad=1)
        ax.set_ylabel(f"PC2 {var[1]:.0f}%", fontsize=7, labelpad=1)
        if n_components == 3:
            ax.set_zlabel(f"PC3 {var[2]:.0f}%", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)

    # hide any unused subplot slots
    for empty in range(k, nrows * ncols):
        fig.add_subplot(nrows, ncols, empty + 1).set_visible(False)

    fig.suptitle("Per-subgraph PCA", fontsize=13)
    plt.tight_layout()
    plt.savefig("imgs/per_subgraph_pca.png")
    plt.close()

# ── Raster + population rate ──────────────────────────────────────────────────

def plot_raster(sol_ss, t_ss, slices, motif_labels=None,
                threshold=0.01):
    k = len(slices)
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(k)]
    motif_colors  = make_palette(k)
    neuron_colors = make_neuron_palette(slices, motif_colors)
    N = sol_ss.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(14, 5 + k * 0.4),
                              gridspec_kw={"height_ratios": [3, 1]})

    # raster
    ax = axes[0]
    for ni in range(N):
        times = t_ss[sol_ss[:, ni] > threshold]
        ax.scatter(times, np.full_like(times, ni),
                   s=1.2, color=neuron_colors[ni],
                   alpha=0.6, rasterized=True)

    for sl in slices[:-1]:
        ax.axhline(sl.stop - 0.5, color="black",
                   lw=0.8, linestyle="--", alpha=0.3)

    ax.set_yticks([sl.start + (sl.stop - sl.start) / 2
                   for sl in slices])
    ax.set_yticklabels(motif_labels, fontsize=9)
    ax.set_ylabel("Neuron", fontsize=10)
    ax.set_xlim(t_ss[0], t_ss[-1])
    ax.set_ylim(-0.5, N - 0.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Raster plot", fontsize=11, loc="left")

    # population rate
    ax2 = axes[1]
    for idx, sl in enumerate(slices):
        pop = sol_ss[:, sl].mean(axis=1)
        ax2.plot(t_ss, pop, color=motif_colors[idx],
                 lw=1.5, label=motif_labels[idx])

    ax2.set_xlabel("Time", fontsize=10)
    ax2.set_ylabel("Mean rate", fontsize=10)
    ax2.set_xlim(t_ss[0], t_ss[-1])

    # legend: inline for small k, outside for large k
    if k <= 6:
        ax2.legend(fontsize=8, loc="upper right", framealpha=0.4)
    else:
        ax2.legend(fontsize=7, loc="upper left",
                   bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("imgs/raster_plot.png")
    plt.close()

# ── Inter-motif phase portrait ────────────────────────────────────────────────

def plot_phase_portrait(sol_ss, t_ss, slices, motif_labels=None,
                        max_pairs=10, ncols=4):
    """
    Pairwise phase portraits of motif mean activities.
    Caps at max_pairs pairs for readability — picks the most
    active pairs by total variance if there are too many.
    """
    k = len(slices)
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(k)]
    motif_colors  = make_palette(k)

    # mean activity per motif
    rates = np.stack([sol_ss[:, sl].mean(axis=1)
                      for sl in slices], axis=1)  # (T, k)

    all_pairs = list(combinations(range(k), 2))

    # if too many pairs, select those with highest joint variance
    if len(all_pairs) > max_pairs:
        pair_var = [(i, j, rates[:, i].var() + rates[:, j].var())
                    for i, j in all_pairs]
        pair_var.sort(key=lambda x: -x[2])
        all_pairs = [(i, j) for i, j, _ in pair_var[:max_pairs]]
        print(f"  Showing top {max_pairs} most active pairs "
              f"(of {k*(k-1)//2} total)")

    n_pairs = len(all_pairs)
    if n_pairs == 0:
        print("  Skipping phase portraits (need ≥ 2 motifs)")
        return
    ncols   = min(ncols, n_pairs)
    nrows   = int(np.ceil(n_pairs / ncols))
    colors_time = plt.cm.viridis(np.linspace(0, 1, len(t_ss)))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4 * nrows),
                              squeeze=False)

    for plot_idx, (i, j) in enumerate(all_pairs):
        row = plot_idx // ncols
        col = plot_idx  % ncols
        ax  = axes[row][col]

        for t in range(len(t_ss) - 1):
            ax.plot(rates[t:t+2, i], rates[t:t+2, j],
                    color=colors_time[t], lw=0.5, alpha=0.7)

        ax.scatter(rates[0,  i], rates[0,  j],
                   color="green", s=40, zorder=5, label="start")
        ax.scatter(rates[-1, i], rates[-1, j],
                   color="red",   s=40, zorder=5, label="end")

        ax.set_xlabel(motif_labels[i], fontsize=9,
                      color=motif_colors[i])
        ax.set_ylabel(motif_labels[j], fontsize=9,
                      color=motif_colors[j])
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    # hide unused slots
    for empty in range(n_pairs, nrows * ncols):
        axes[empty // ncols][empty % ncols].set_visible(False)

    fig.suptitle("Inter-motif phase portraits", fontsize=13)
    plt.tight_layout()
    plt.savefig("imgs/inter_motif_phase_portraits.png")
    plt.close()

# ── UMAP ──────────────────────────────────────────────────────────────────────

def plot_umap(sol_ss, slices, motif_labels=None):
    try:
        import umap
    except ImportError:
        print("  UMAP not available — run: pip install umap-learn")
        return

    k = len(slices)
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(k)]
    motif_colors = make_palette(k)

    reducer   = umap.UMAP(n_components=2, n_neighbors=30,
                           min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(sol_ss)

    rates    = np.stack([sol_ss[:, sl].mean(axis=1)
                         for sl in slices], axis=1)
    dominant = rates.argmax(axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx in range(k):
        mask = dominant == idx
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   s=2, alpha=0.5,
                   color=motif_colors[idx],
                   label=motif_labels[idx],
                   rasterized=True)

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

    if k <= 8:
        ax.legend(fontsize=9, markerscale=5)
    else:
        ax.legend(fontsize=7, markerscale=5,
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.set_title("UMAP — colored by dominant motif", fontsize=11)
    plt.tight_layout()
    plt.savefig("imgs/umap.png")
    plt.close()

# ── Master function ───────────────────────────────────────────────────────────

def full_visualization(sol, t, slices, motif_labels=None,
                       transient_frac=0.5, color_by_motif=0,
                       ncols=4, max_pairs=10,
                       show_pca=True, show_raster=True,
                       show_phase=True, show_umap=True):
    """
    Full visualization suite — scales to any number of motifs.

    Parameters
    ----------
    ncols        : columns in PCA and phase portrait grids
    max_pairs    : cap on pairwise phase portraits shown
    show_*       : toggle individual panels on/off
    """
    k = len(slices)
    N = sol.shape[1]
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(k)]

    cut    = int(len(t) * transient_frac)
    t_ss   = t[cut:]
    sol_ss = sol[cut:]

    print(f"Network: {k} motifs, {N} neurons total")
    print(f"Steady state: t={t_ss[0]:.1f} to {t_ss[-1]:.1f}")
    for idx, sl in enumerate(slices):
        std  = sol_ss[:, sl].std(axis=0).mean()
        mean = sol_ss[:, sl].mean()
        tag  = "oscillating" if std > 0.01 else "static"
        print(f"  {motif_labels[idx]:15s} | "
              f"mean={mean:.4f}  std={std:.4f}  {tag}")

    if show_pca:
        print("\n── Per-subgraph PCA ─────────────────────────────────")
        plot_per_subgraph_pca(sol_ss, slices, motif_labels,
                              color_by_motif=color_by_motif,
                              ncols=ncols)
    if show_raster:
        print("── Raster + population rate ─────────────────────────")
        plot_raster(sol_ss, t_ss, slices, motif_labels)

    if show_phase:
        print("── Inter-motif phase portraits ───────────────────────")
        plot_phase_portrait(sol_ss, t_ss, slices, motif_labels,
                            max_pairs=max_pairs, ncols=ncols)
    if show_umap:
        print("── UMAP embedding ───────────────────────────────────")
        plot_umap(sol_ss, slices, motif_labels)

# # Build network
# eps   = 0.10
# delta = 0.50
# theta = 0.10
# dc    = 0.05   # coexisting regime

# W, slices = build_gctln([3, 3], eps=eps, delta=delta,
#                          delta_cross=dc)

# x0 = np.array([0.3, 0.2, 0.1,
#                 0.3, 0.2, 0.1])

# # Simulate
# t, sol = simulate(W, theta, T=500, n_steps=15000, x0=x0)

# # Visualize
# full_visualization(
#     sol          = sol,
#     t            = t,
#     slices       = slices,
#     motif_labels = ["CPG-A", "CPG-B"],
#     transient_frac = 0.5,    # discard first 50% as transient
#     color_by_motif = 1       # color CPG-A's PCA trajectory by CPG-B's activity
# )

# # 2 motifs of 3 neurons
# W, slices = build_gctln([3, 3], eps=0.1, delta=0.5, delta_cross=0.05)
# t, sol    = simulate(W, theta=0.1, T=500, n_steps=15000,
#                      x0=np.random.uniform(0.05, 0.4, 6))
# full_visualization(sol, t, slices,
#                    motif_labels=["CPG-A", "CPG-B"])

# # 5 heterogeneous motifs
# W, slices = build_gctln([3, 3, 4, 4, 5],
#                          eps=0.1, delta=0.5, delta_cross=0.05)
# t, sol    = simulate(W, theta=0.1, T=800, n_steps=20000,
#                      x0=np.random.uniform(0.05, 0.4, 19))
# full_visualization(sol, t, slices,
#                    motif_labels=["V1","V2","Motor","Timing","Memory"],
#                    ncols=3, max_pairs=8)

# # turn off UMAP if not installed
# full_visualization(sol, t, slices, show_umap=False)


# W, slices = build_gctln([3, 3], eps=0.1, delta=0.5, delta_cross=0.05)

# # In phase — identical to symmetric x0 behavior
# x0_inphase = np.array([0.3, 0.2, 0.1,
#                         0.3, 0.2, 0.1])

# # Out of phase — A is ahead of B by one step in the cycle
# x0_outphase = np.array([0.3, 0.2, 0.1,
#                          0.1, 0.3, 0.2])

# # Maximally out of phase — A is halfway around the cycle from B
# x0_halfphase = np.array([0.3, 0.2, 0.1,
#                           0.1, 0.2, 0.3])

# for label, x0 in [("in phase",    x0_inphase),
#                    ("out of phase", x0_outphase),
#                    ("half phase",   x0_halfphase)]:
#     t, sol = simulate(W, theta=0.10, T=500, n_steps=15000, x0=x0)
#     cut    = int(0.5 * len(t))
#     sol_ss = sol[cut:]
#     pca    = PCA(n_components=2).fit_transform(sol_ss)
#     var    = PCA(n_components=2).fit(sol_ss).explained_variance_ratio_ * 100
#     print(f"{label:15s} | "
#           f"std_A={sol_ss[:, slices[0]].std(axis=0).mean():.4f} "
#           f"std_B={sol_ss[:, slices[1]].std(axis=0).mean():.4f} | "
#           f"traj range: {(sol_ss.max(axis=0) - sol_ss.min(axis=0)).mean():.4f}")

## idrk if this is a helpful visual

from scipy.signal import hilbert

def extract_phase(signal):
    """
    Extract instantaneous phase of an oscillatory signal
    using the Hilbert transform.
    Returns phase in [0, 2pi).
    """
    analytic  = hilbert(signal - signal.mean())
    phase     = np.angle(analytic) % (2 * np.pi)
    return phase

def plot_torus_structure(sol_ss, t_ss, slices, motif_labels=None):
    """
    Three panels:
    1. Phase of each motif over time — shows frequency and drift
    2. Phase difference — constant = locked, drifting = quasiperiodic
    3. Phase portrait on the torus (phi_A vs phi_B) — 
       closed curve = rational ratio, dense fill = irrational ratio
    """
    k = len(slices)
    if k < 2:
        print("  Skipping torus structure (need ≥ 2 motifs)")
        return
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(k)]
    motif_colors = make_palette(k)

    # Use first neuron of each motif as the representative oscillator
    # (all neurons in a motif are phase-shifted versions of each other)
    phases = []
    for sl in slices:
        sig   = sol_ss[:, sl.start]
        phase = extract_phase(sig)
        phases.append(phase)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: raw phase over time
    ax = axes[0]
    for idx, phase in enumerate(phases):
        ax.plot(t_ss, phase,
                color=motif_colors[idx],
                lw=0.8, alpha=0.7,
                label=motif_labels[idx])
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Phase (radians)", fontsize=10)
    ax.set_title("Instantaneous phase", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    # Panel 2: phase differences between all pairs
    ax2 = axes[1]
    for i, j in combinations(range(k), 2):
        diff = phases[i] - phases[j]
        # unwrap to show drift continuously
        diff_unwrapped = np.unwrap(diff)
        ax2.plot(t_ss, diff_unwrapped,
                 lw=0.8, alpha=0.8,
                 label=f"{motif_labels[i]} - {motif_labels[j]}")
    ax2.set_xlabel("Time", fontsize=10)
    ax2.set_ylabel("Phase difference", fontsize=10)
    ax2.set_title("Phase drift\n(flat=locked, slope=quasiperiodic)",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines[["top","right"]].set_visible(False)

    # Panel 3: torus portrait — phi_A vs phi_B
    # For k=2 this is a 2D torus slice
    # For k>2 show the first two motifs
    ax3 = axes[2]
    colors_time = plt.cm.viridis(np.linspace(0, 1, len(t_ss)))
    for i in range(len(t_ss) - 1):
        ax3.plot(phases[0][i:i+2], phases[1][i:i+2],
                 color=colors_time[i], lw=0.4, alpha=0.6)

    ax3.set_xlabel(f"Phase: {motif_labels[0]}", fontsize=10)
    ax3.set_ylabel(f"Phase: {motif_labels[1]}", fontsize=10)
    ax3.set_title("Torus portrait (φ_A vs φ_B)\n"
                  "closed curve=locked, filled=quasiperiodic",
                  fontsize=11)
    ax3.set_xlim(0, 2*np.pi)
    ax3.set_ylim(0, 2*np.pi)
    ax3.set_xticks([0, np.pi, 2*np.pi])
    ax3.set_xticklabels(["0", "π", "2π"])
    ax3.set_yticks([0, np.pi, 2*np.pi])
    ax3.set_yticklabels(["0", "π", "2π"])
    ax3.spines[["top","right"]].set_visible(False)
    ax3.set_aspect("equal")

    plt.suptitle("Torus structure of coupled oscillators", fontsize=13)
    plt.tight_layout()
    plt.close("all")

    # Print frequency ratio
    # Estimate frequency from mean phase velocity
    freqs = []
    for phase in phases:
        dphi = np.diff(np.unwrap(phase))
        freq = dphi.mean() / (t_ss[1] - t_ss[0]) / (2 * np.pi)
        freqs.append(freq)
        
    print("\nEstimated frequencies:")
    for idx, f in enumerate(freqs):
        print(f"  {motif_labels[idx]}: {f:.4f} Hz")
    if len(freqs) >= 2:
        ratio = freqs[0] / freqs[1]
        print(f"\nFrequency ratio f_A/f_B = {ratio:.4f}")
        # Check proximity to simple rational numbers
        from fractions import Fraction
        frac = Fraction(ratio).limit_denominator(10)
        print(f"Closest simple rational: {frac} = {float(frac):.4f}")
        if abs(ratio - float(frac)) < 0.01:
            print(f"  → Phase locked {frac.numerator}:{frac.denominator}")
        else:
            print(f"  → Quasiperiodic (irrational ratio)")

# W, slices = build_gctln([3, 3], eps=0.1, delta=0.5, delta_cross=0.05)
# t, sol    = simulate(W, theta=0.1, T=1000, n_steps=30000,
#                      x0=np.random.uniform(0.05, 0.4, 6))

# cut    = int(0.4 * len(t))
# t_ss   = t[cut:]
# sol_ss = sol[cut:]

# plot_torus_structure(sol_ss, t_ss, slices,
#                      motif_labels=["CPG-A", "CPG-B"])

# Torus projection

def plot_torus_3d(sol_ss, t_ss, slices, motif_labels=None,
                  R=3.0, r=1.0, surface_alpha=0.08,
                  n_surface=100, fade_window=None):
    """
    Plot the trajectory of two coupled oscillators on a 3D torus.

    Parameters
    ----------
    R             : major radius (hole size) — increase to spread the torus
    r             : minor radius (tube thickness)
    surface_alpha : transparency of the torus shell (0=invisible, 1=solid)
    n_surface     : resolution of the surface mesh
    fade_window   : if set, only show this many recent steps as a trail
                    None shows the full trajectory
    """
    if len(slices) < 2:
        print("  Skipping torus plot (need ≥ 2 motifs)")
        return
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(len(slices))]

    # ── Extract phases ────────────────────────────────────────────────────
    phase_A = extract_phase(sol_ss[:, slices[0].start])
    phase_B = extract_phase(sol_ss[:, slices[1].start])

    # ── Map phases to torus coordinates ──────────────────────────────────
    def torus_coords(phi_A, phi_B, R, r):
        x = (R + r * np.cos(phi_B)) * np.cos(phi_A)
        y = (R + r * np.cos(phi_B)) * np.sin(phi_A)
        z = r * np.sin(phi_B)
        return x, y, z

    traj_x, traj_y, traj_z = torus_coords(phase_A, phase_B, R, r)

    # ── Torus surface mesh ────────────────────────────────────────────────
    u = np.linspace(0, 2*np.pi, n_surface)
    v = np.linspace(0, 2*np.pi, n_surface)
    U, V = np.meshgrid(u, v)
    Sx, Sy, Sz = torus_coords(U, V, R, r)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Torus shell — very transparent so trajectory is visible
    ax.plot_surface(Sx, Sy, Sz,
                    color="steelblue",
                    alpha=surface_alpha,
                    linewidth=0,
                    antialiased=True,
                    zorder=0)

    # Torus wireframe for structure
    ax.plot_wireframe(Sx, Sy, Sz,
                      color="steelblue",
                      alpha=0.06,
                      linewidth=0.3,
                      zorder=1)

    # ── Trajectory ───────────────────────────────────────────────────────
    n = len(traj_x)
    if fade_window is not None:
        # Only plot recent history, fading older segments
        start = max(0, n - fade_window)
        xs = traj_x[start:]
        ys = traj_y[start:]
        zs = traj_z[start:]
        alphas = np.linspace(0.05, 0.9, len(xs))
        colors_t = plt.cm.plasma(np.linspace(0.2, 0.9, len(xs)))
        for i in range(len(xs)-1):
            ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                    color=colors_t[i],
                    lw=1.2, alpha=alphas[i], zorder=3)
    else:
        # Full trajectory colored by time
        colors_t = plt.cm.plasma(np.linspace(0.2, 0.9, n))
        for i in range(n-1):
            ax.plot(traj_x[i:i+2],
                    traj_y[i:i+2],
                    traj_z[i:i+2],
                    color=colors_t[i],
                    lw=0.8, alpha=0.6, zorder=3)

    # Mark start and current end
    ax.scatter(*torus_coords(phase_A[0],  phase_B[0],  R, r),
               color="lime",  s=60, zorder=5, label="start")
    ax.scatter(*torus_coords(phase_A[-1], phase_B[-1], R, r),
               color="red",   s=60, zorder=5, label="end")

    # ── Axes and labels ───────────────────────────────────────────────────
    ax.set_xlabel("X", fontsize=9, labelpad=1)
    ax.set_ylabel("Y", fontsize=9, labelpad=1)
    ax.set_zlabel("Z", fontsize=9, labelpad=1)
    ax.tick_params(labelsize=7)
    ax.set_box_aspect([1, 1, 0.5])
    ax.legend(fontsize=9, loc="upper left")

    lim = R + r + 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-r - 0.5, r + 0.5)

    ax.set_title(
        f"Trajectory on torus  |  "
        f"φ_big = {motif_labels[0]},  "
        f"φ_tube = {motif_labels[1]}\n"
        f"color = time (purple → yellow)",
        fontsize=11
    )

    plt.tight_layout()
    plt.close("all")


def animate_torus_save(sol_ss, t_ss, slices, motif_labels=None,
                       R=3.0, r=1.0, surface_alpha=0.07,
                       n_surface=80, fade_window=150,
                       interval=30, n_frames=2000,
                       filename="vids/torus_animation.mp4",
                       fps=30, dpi=120):
    """
    Same as animate_torus but saves directly to MP4.
    Much faster and lighter than to_jshtml for long animations.
    """
    if len(slices) < 2:
        print("  Skipping torus animation (need ≥ 2 motifs)")
        return
    if motif_labels is None:
        motif_labels = [f"Motif {i}" for i in range(len(slices))]

    phase_A = extract_phase(sol_ss[:, slices[0].start])
    phase_B = extract_phase(sol_ss[:, slices[1].start])

    def torus_coords(phi_A, phi_B):
        x = (R + r * np.cos(phi_B)) * np.cos(phi_A)
        y = (R + r * np.cos(phi_B)) * np.sin(phi_A)
        z = r * np.sin(phi_B)
        return x, y, z

    # Downsample
    total   = len(phase_A)
    indices = np.linspace(0, total - 1, n_frames).astype(int)
    phase_A = phase_A[indices]
    phase_B = phase_B[indices]

    traj_x, traj_y, traj_z = torus_coords(phase_A, phase_B)

    # Surface
    u, v       = np.linspace(0, 2*np.pi, n_surface), \
                 np.linspace(0, 2*np.pi, n_surface)
    U, V       = np.meshgrid(u, v)
    Sx, Sy, Sz = torus_coords(U, V)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot_surface(Sx, Sy, Sz, color="steelblue",
                    alpha=surface_alpha, linewidth=0,
                    antialiased=True, zorder=0)
    ax.plot_wireframe(Sx, Sy, Sz, color="steelblue",
                      alpha=0.05, linewidth=0.3, zorder=1)

    lim = R + r + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-r - 0.3, r + 0.3)
    ax.set_box_aspect([1, 1, 0.5])
    ax.tick_params(labelsize=6)
    ax.set_title(
        f"φ_big={motif_labels[0]},  φ_tube={motif_labels[1]}  "
        f"({n_frames} frames, {n_frames/fps:.0f}s)",
        fontsize=11)

    trail, = ax.plot([], [], [], lw=1.0, color="mediumpurple",
                     alpha=0.7, zorder=3)
    point, = ax.plot([], [], [], "o", markersize=7,
                     color="yellow",
                     markeredgecolor="black",
                     markeredgewidth=0.8, zorder=5)

    # Progress printing — useful for long renders
    def update(frame):
        if frame % 200 == 0:
            print(f"  Rendering frame {frame}/{n_frames} "
                  f"({100*frame/n_frames:.0f}%)", end="\r")
        start = max(0, frame - fade_window)
        trail.set_data(traj_x[start:frame],
                       traj_y[start:frame])
        trail.set_3d_properties(traj_z[start:frame])
        point.set_data([traj_x[frame]], [traj_y[frame]])
        point.set_3d_properties([traj_z[frame]])
        return trail, point

    ani = FuncAnimation(fig, update, frames=n_frames,
                        interval=interval, blit=True)

    print(f"Saving {filename} — {n_frames} frames at {fps}fps "
          f"({n_frames/fps:.0f}s)...")
    ani.save(filename, writer="ffmpeg", fps=fps, dpi=dpi,
             extra_args=["-vcodec", "libx264",
                         "-pix_fmt", "yuv420p"])
    plt.close(fig)
    print(f"\nSaved: {filename}")
    return ani


if __name__ == "__main__":
    # ── Run ───────────────────────────────────────────────────────────────────────
    # W, slices = build_gctln([3, 3], eps=0.1, delta=0.5, delta_cross=0.05)
    # t, sol    = simulate(W, theta=0.1, T=1000, n_steps=300000,
    #                      x0=np.random.uniform(0.05, 0.4, 6))

    # cut    = int(0.3 * len(t))
    # t_ss   = t[cut:]
    # sol_ss = sol[cut:]

    # # Static torus with full trajectory
    # plot_torus_3d(sol_ss, t_ss, slices,
    #               motif_labels=["CPG-A", "CPG-B"],
    #               R=3.0, r=1.0,
    #               surface_alpha=0.08)

    # animate_torus_save(
    #     sol_ss, t_ss, slices,
    #     motif_labels  = ["CPG-A", "CPG-B"],
    #     R             = 3.0,
    #     r             = 1.0,
    #     surface_alpha = 0.08,
    #     fade_window   = 150,
    #     n_frames      = 3000,    # 100 seconds at 30fps
    #     fps           = 30,
    #     dpi           = 120,
    #     filename      = "torus_cpg.mp4"
    # )
    pass