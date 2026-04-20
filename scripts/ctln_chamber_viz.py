"""
ctln_chamber_viz.py
-------------------
Visualize a CTLN limit cycle threading through linear regimes in state space.

The goal is to make visible the fact that the switching hyperplanes
x_i = 0 are not merely geometric surfaces but boundaries between
*entirely different linear systems*. As the trajectory crosses each
hyperplane, the active set sigma changes, the Jacobian jumps, and
the local vector field is governed by a different matrix.

Drop-in compatible with gctln_* module style:
  - build_ctln_W(G, eps, delta) produces W from a directed graph
  - simulate_ctln(...) uses scipy.integrate.solve_ivp
  - render_chamber_figure(...) makes the figure

Run standalone:
  python ctln_chamber_viz.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp
from itertools import combinations
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Core CTLN construction and simulation
# --------------------------------------------------------------------------

def build_ctln_W(adj: np.ndarray, eps: float, delta: float) -> np.ndarray:
    """
    Build the CTLN weight matrix W from a directed adjacency matrix.

    adj[i, j] == 1 means there is an edge j -> i (Curto convention).
    W_ij = 0            if i == j
         = -1 + eps     if j -> i   (edge present)
         = -1 - delta   otherwise   (no edge)
    """
    n = adj.shape[0]
    W = np.where(adj == 1, -1.0 + eps, -1.0 - delta)
    np.fill_diagonal(W, 0.0)
    return W


def ctln_rhs(t: float, x: np.ndarray, W: np.ndarray, theta: float) -> np.ndarray:
    """dx/dt = -x + ReLU(W x + theta)"""
    return -x + np.maximum(W @ x + theta, 0.0)


def simulate_ctln(
    W: np.ndarray,
    theta: float,
    x0: np.ndarray,
    t_span: tuple[float, float] = (0.0, 120.0),
    t_eval_n: int = 6000,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """Integrate the CTLN with a tight tolerance so crossings are clean."""
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_n)
    sol = solve_ivp(
        ctln_rhs,
        t_span,
        x0,
        args=(W, theta),
        method="LSODA",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    return sol.t, sol.y  # y has shape (n, T)


# --------------------------------------------------------------------------
# Active-set / chamber labeling
# --------------------------------------------------------------------------

def active_set_trace(
    X: np.ndarray, W: np.ndarray, theta: float, tol: float = 1e-6
) -> np.ndarray:
    """
    For each time step, return the active set sigma as a bitmask over n neurons.

    A neuron i is 'active' in the piecewise-linear sense when the ReLU
    argument (W x + theta)_i > 0. This is the definition that actually
    governs which linear system is running: inside a chamber where sigma
    is constant, dx/dt = -x + W_sigma x + theta_sigma is linear.
    """
    n = W.shape[0]
    arg = W @ X + theta  # shape (n, T)
    active = (arg > tol).astype(np.int8)
    # Bitmask encoding: neuron 0 is LSB
    weights = (1 << np.arange(n)).reshape(-1, 1)
    return (active * weights).sum(axis=0)


def bitmask_to_tuple(mask: int, n: int) -> tuple[int, ...]:
    """Decode a bitmask back into a sorted tuple of active neuron indices."""
    return tuple(i for i in range(n) if (mask >> i) & 1)


def sigma_label(mask: int, n: int) -> str:
    """Human-readable label like '{1,2}' (1-indexed to match CTLN literature)."""
    idx = bitmask_to_tuple(mask, n)
    if not idx:
        return "∅"
    return "{" + ",".join(str(i + 1) for i in idx) + "}"


def detect_crossings(sigma_trace: np.ndarray) -> list[int]:
    """Indices where the active set changes (chamber crossings)."""
    return list(np.where(np.diff(sigma_trace) != 0)[0] + 1)


# --------------------------------------------------------------------------
# Local linear system per chamber (the 'different regime' evidence)
# --------------------------------------------------------------------------

@dataclass
class ChamberRegime:
    """The linear system that is active in chamber sigma."""
    sigma: tuple[int, ...]
    n: int
    W: np.ndarray
    theta: float

    @property
    def jacobian(self) -> np.ndarray:
        """
        Inside chamber sigma, dx/dt = -x + W_sigma x + theta_sigma,
        so the Jacobian is -I + W restricted to active rows/columns.
        Rows/columns not in sigma get only the -I contribution (silent
        neurons decay toward 0 independently).
        """
        J = -np.eye(self.n)
        for i in self.sigma:
            for j in self.sigma:
                J[i, j] += self.W[i, j]
        return J

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.linalg.eigvals(self.jacobian)

    def fixed_point(self) -> np.ndarray | None:
        """
        The 'virtual' fixed point of this linear system, computed by
        solving the restricted linear system. Returns None if singular.
        Note: this fixed point may lie outside the chamber sigma, in
        which case it is not a true fixed point of the CTLN.
        """
        if not self.sigma:
            return np.zeros(self.n)
        idx = list(self.sigma)
        A = -np.eye(len(idx)) + self.W[np.ix_(idx, idx)]
        try:
            x_sigma = np.linalg.solve(A, -self.theta * np.ones(len(idx)))
        except np.linalg.LinAlgError:
            return None
        fp = np.zeros(self.n)
        for k, i in enumerate(idx):
            fp[i] = x_sigma[k]
        return fp


# --------------------------------------------------------------------------
# Geometry helpers for 3D rendering of hyperplanes in the positive orthant
# --------------------------------------------------------------------------

def interior_hyperplane_polygon(
    W: np.ndarray, theta: float, i: int, box_max: float
) -> np.ndarray | None:
    """
    Compute the polygon where neuron i's interior nullcline hyperplane
    (sum_j W_ij x_j + theta = 0) intersects the box [0, box_max]^3.

    Returns vertices in cyclic order, or None if the plane does not
    intersect the positive orthant box.
    """
    n = W.shape[0]
    assert n == 3, "3D polygon helper assumes n=3"
    # Plane: a . x + theta = 0  =>  a . x = -theta
    a = W[i].copy()

    # Enumerate edges of the box and keep intersection points.
    verts = []
    for dim in range(3):
        other = [d for d in range(3) if d != dim]
        for v1 in (0.0, box_max):
            for v2 in (0.0, box_max):
                # Parameterize edge along dim from 0 to box_max
                # with the other two coords fixed at (v1, v2).
                # Solve a . x = -theta for x[dim].
                a_dim = a[dim]
                if abs(a_dim) < 1e-12:
                    continue
                fixed_sum = a[other[0]] * v1 + a[other[1]] * v2
                x_dim = (-theta - fixed_sum) / a_dim
                if 0.0 <= x_dim <= box_max:
                    point = np.zeros(3)
                    point[dim] = x_dim
                    point[other[0]] = v1
                    point[other[1]] = v2
                    verts.append(point)

    if len(verts) < 3:
        return None

    verts = np.array(verts)
    # Deduplicate
    verts = np.unique(np.round(verts, 8), axis=0)
    if len(verts) < 3:
        return None

    # Sort vertices around their centroid so the polygon is non-self-intersecting
    centroid = verts.mean(axis=0)
    # Project to 2D using the plane's two dominant axes
    normal = a / np.linalg.norm(a)
    # Pick two in-plane basis vectors
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rel = verts - centroid
    angles = np.arctan2(rel @ v, rel @ u)
    order = np.argsort(angles)
    return verts[order]


# --------------------------------------------------------------------------
# Fixed point enumeration
# --------------------------------------------------------------------------

def all_fixed_points(
    W: np.ndarray, theta: float, tol: float = 1e-8
) -> list[dict]:
    """
    Enumerate all 2^n linear-system fixed points, classify each as
    'true' (lies in its own chamber) or 'virtual' (lies outside).

    Returns a list of dicts with keys:
      sigma:      tuple of active neuron indices
      point:      np.ndarray of shape (n,), the linear fixed point
      in_chamber: bool, True iff the point actually lies in R_sigma
      stability:  'stable' | 'unstable' | 'saddle'
      eigvals:    np.ndarray of Jacobian eigenvalues at this fixed point

    A fixed point with in_chamber=True is a *true* CTLN fixed point.
    A fixed point with in_chamber=False is *virtual* — the linear
    regime's attractor/repeller that the trajectory is "aimed at"
    while inside R_sigma, even though the point itself is outside.
    """
    n = W.shape[0]
    results = []
    for r in range(0, n + 1):
        for sigma in combinations(range(n), r):
            reg = ChamberRegime(sigma=sigma, n=n, W=W, theta=theta)
            fp = reg.fixed_point()
            if fp is None:
                continue
            # Check whether fp lies in R_sigma, i.e. whether exactly the
            # neurons in sigma are active at fp.
            arg = W @ fp + theta
            active_at_fp = set(i for i in range(n) if arg[i] > tol)
            in_chamber = active_at_fp == set(sigma)

            evs = reg.eigenvalues
            max_real = evs.real.max()
            min_real = evs.real.min()
            if max_real < -tol:
                stab = "stable"
            elif min_real > tol:
                stab = "unstable"
            else:
                stab = "saddle"

            results.append({
                "sigma": sigma,
                "point": fp,
                "in_chamber": in_chamber,
                "stability": stab,
                "eigvals": evs,
            })
    return results


# --------------------------------------------------------------------------
# Vector-field hairs on the axis-piece hyperplanes
# --------------------------------------------------------------------------

def _vector_field_hairs_on_face(
    W: np.ndarray,
    theta: float,
    face_axis: int,
    box_max: float,
    n_grid: int = 6,
    hair_length: float = 0.07,
):
    """
    Sample the CTLN vector field on the orthant face { x : x_{face_axis} = 0 }
    on an n_grid x n_grid grid spanning [0, box_max] in the other two coords.

    Returns a list of (start, end, sigma) tuples for each hair, where:
      - start: 3D point on the face
      - end:   3D endpoint (start + normalized velocity * hair_length * box_max)
      - sigma: tuple of active neurons at 'start' (for coloring)

    The hair lives in the *full* 3D space — it points in the direction of
    the true 3D vector field, not just its projection onto the face. This
    makes the flow-through-boundary behavior immediately visible: hairs
    that point away from the face indicate that the trajectory is leaving
    it (neuron face_axis turning on), hairs pointing toward the face
    indicate the trajectory is approaching it.
    """
    n = W.shape[0]
    other_axes = [a for a in range(n) if a != face_axis]
    # Sample on an interior grid — avoid the corners where two faces meet
    # (flow direction degenerates there).
    pad = 0.08 * box_max
    coords = np.linspace(pad, box_max - pad, n_grid)

    hairs = []
    for u in coords:
        for v in coords:
            x = np.zeros(n)
            x[other_axes[0]] = u
            x[other_axes[1]] = v
            # x[face_axis] stays 0

            # Compute dx/dt at this point
            dxdt = -x + np.maximum(W @ x + theta, 0.0)
            speed = np.linalg.norm(dxdt)
            if speed < 1e-9:
                continue

            # Normalize to fixed hair length
            direction = dxdt / speed
            start = x.copy()
            end = x + direction * hair_length * box_max

            # Active set at this base point
            arg = W @ x + theta
            sigma = tuple(i for i in range(n) if arg[i] > 1e-9)
            hairs.append((start, end, sigma))

    return hairs


def _draw_vector_field_hairs(ax, W, theta, box_max, n_grid=5):
    """
    Draw thin line-segment hairs on all three orthant faces, colored by
    active-set chamber at each basepoint. A small filled tick at the tip
    indicates direction.
    """
    n = W.shape[0]
    # Collect all hairs first so we can compute a shared speed scale,
    # which lets us filter out near-stagnant samples without them
    # dominating the figure's visual density.
    all_hairs = []
    for face_axis in range(n):
        hairs = _vector_field_hairs_on_face(
            W, theta, face_axis, box_max, n_grid=n_grid,
            hair_length=0.09,
        )
        all_hairs.extend(hairs)

    # Compute speed for each (via length of (end - start) vector normalized back out)
    # and use it to modulate alpha so stronger flow reads more prominently.
    # Since hairs were normalized to fixed length, we need to recompute speed
    # from the original vector field. Do it here for weighting purposes.
    speeds = []
    for start, end, _ in all_hairs:
        x = start
        dxdt = -x + np.maximum(W @ x + theta, 0.0)
        speeds.append(np.linalg.norm(dxdt))
    if not speeds:
        return
    speeds = np.array(speeds)
    # Normalize so the fastest hair is alpha=0.9, slowest is alpha=0.3.
    max_speed = max(speeds.max(), 1e-6)

    for (start, end, sigma), speed in zip(all_hairs, speeds):
        col = chamber_color(sigma)
        alpha = 0.3 + 0.6 * (speed / max_speed)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=col, linewidth=0.85, alpha=alpha,
            solid_capstyle="round", zorder=3,
        )
        # Filled dot at the tip to indicate direction of flow
        ax.scatter(
            [end[0]], [end[1]], [end[2]],
            color=col, s=7, alpha=alpha,
            depthshade=False, zorder=3, edgecolors="none",
        )


# --------------------------------------------------------------------------
# Main rendering routine
# --------------------------------------------------------------------------

# Palette — muted, print-friendly; matches the common CTLN-paper aesthetic.
_CHAMBER_COLORS = {
    (): "#cccccc",
    (0,): "#c0dd97",
    (1,): "#f5c4b3",
    (2,): "#b5d4f4",
    (0, 1): "#d4537e",   # 1,2 active
    (1, 2): "#534ab7",   # 2,3 active
    (0, 2): "#1d9e75",   # 1,3 active
    (0, 1, 2): "#5f5e5a",
}


def chamber_color(sigma: tuple[int, ...]) -> str:
    return _CHAMBER_COLORS.get(sigma, "#888888")


def render_chamber_figure(
    W: np.ndarray,
    theta: float,
    t: np.ndarray,
    X: np.ndarray,
    sigma_trace: np.ndarray,
    crossings: list[int],
    *,
    box_max: float | None = None,
    title: str = "CTLN limit cycle threading chambers",
    savepath: str | None = None,
    show_vector_field: bool = True,
    vector_field_grid: int = 5,
):
    """
    Master rendering function. Produces a 3-panel figure:
      (left)  3D state space with hyperplanes and trajectory colored by sigma
      (top right) time series of x_i(t), shaded by active chamber
      (bot right) table of chamber regimes visited, with Jacobian eigenvalues

    If show_vector_field is True, thin vector-field hairs are drawn on the
    three axis-piece hyperplanes (orthant faces x_i = 0), revealing the
    local flow direction at each switching boundary.
    """
    n = W.shape[0]
    assert n == 3, "This visualization is specialized for n=3"

    if box_max is None:
        box_max = float(np.ceil(X.max() * 1.1))

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.3, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.35,
        wspace=0.25,
        top=0.88,
    )

    # ---------- (A) 3D state space ----------
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    _draw_3d_panel(
        ax3d, W, theta, X, sigma_trace, crossings, box_max,
        show_vector_field=show_vector_field,
        vector_field_grid=vector_field_grid,
    )

    # ---------- (B) time series ----------
    ax_ts = fig.add_subplot(gs[0, 1])
    _draw_timeseries_panel(ax_ts, t, X, sigma_trace, crossings)

    # ---------- (C) regime table ----------
    ax_reg = fig.add_subplot(gs[1, 1])
    _draw_regime_panel(ax_reg, W, theta, sigma_trace, crossings, n)

    fig.suptitle(title, fontsize=13, y=0.98)
    if savepath:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    return fig


def _draw_3d_panel(
    ax, W, theta, X, sigma_trace, crossings, box_max,
    *, show_vector_field=True, vector_field_grid=6,
):
    n = W.shape[0]

    # Interior hyperplanes (one per neuron) — these are the "different regime" surfaces
    plane_colors = ["#185fa5", "#3b6d11", "#993556"]
    plane_labels = ["N₁: W₁·x + θ = 0", "N₂: W₂·x + θ = 0", "N₃: W₃·x + θ = 0"]
    for i in range(n):
        poly = interior_hyperplane_polygon(W, theta, i, box_max)
        if poly is not None and len(poly) >= 3:
            pc = Poly3DCollection(
                [poly],
                alpha=0.18,
                facecolor=plane_colors[i],
                edgecolor=plane_colors[i],
                linewidth=1.2,
            )
            ax.add_collection3d(pc)

    # Axis-piece nullclines (the switching boundaries x_i = 0) as wireframe hints.
    # Each is a face of the positive orthant box.
    axis_colors = ["#185fa5", "#3b6d11", "#993556"]
    faces = [
        # x1 = 0 face
        np.array([[0, 0, 0], [0, box_max, 0], [0, box_max, box_max], [0, 0, box_max]]),
        # x2 = 0 face
        np.array([[0, 0, 0], [box_max, 0, 0], [box_max, 0, box_max], [0, 0, box_max]]),
        # x3 = 0 face
        np.array([[0, 0, 0], [box_max, 0, 0], [box_max, box_max, 0], [0, box_max, 0]]),
    ]
    for face, col in zip(faces, axis_colors):
        pc = Poly3DCollection(
            [face], alpha=0.04, facecolor=col, edgecolor=col,
            linewidth=0.5, linestyle="--",
        )
        ax.add_collection3d(pc)

    # Vector-field hairs on the axis-piece hyperplanes. These reveal local
    # flow direction at the switching boundaries; hairs on either side of
    # a crossing point in visibly different directions, making the
    # regime shift readable from the field alone.
    if show_vector_field:
        _draw_vector_field_hairs(ax, W, theta, box_max, n_grid=vector_field_grid)

    # Trajectory colored by active set sigma
    # We draw segments between consecutive samples, coloring each segment
    # by the sigma of its starting point.
    for k in range(len(sigma_trace) - 1):
        s = bitmask_to_tuple(int(sigma_trace[k]), n)
        col = chamber_color(s)
        ax.plot(
            X[0, k:k + 2], X[1, k:k + 2], X[2, k:k + 2],
            color=col, linewidth=1.6, solid_capstyle="round",
        )

    # Mark chamber crossings
    for idx in crossings:
        ax.scatter(
            X[0, idx], X[1, idx], X[2, idx],
            color="#d85a30", s=42, edgecolor="#4a1b0c", linewidth=0.8,
            depthshade=False, zorder=10,
        )

    # Mark fixed points — both true CTLN fixed points (prominent X) and
    # virtual fixed points per chamber (small open markers).
    fps = all_fixed_points(W, theta)
    for fp in fps:
        pt = fp["point"]
        # Only draw points that fall inside the viewing box so the axes
        # don't blow out; for visible virtual fixed points this filters
        # out anything far outside the orthant.
        if np.any(pt < -0.05 * box_max) or np.any(pt > 1.3 * box_max):
            continue

        if fp["in_chamber"]:
            # True CTLN fixed point: large X, bold, labeled.
            ax.scatter(
                pt[0], pt[1], pt[2],
                marker="X", s=180,
                color="#2c2c2a", edgecolor="#ffffff", linewidth=1.4,
                depthshade=False, zorder=20,
            )
            label = (
                f"CTLN fixed point\n"
                f"σ = {sigma_label(_mask_from_tuple(fp['sigma']), n)}\n"
                f"({fp['stability']})"
            )
            ax.text(
                pt[0], pt[1], pt[2] + 0.06 * box_max,
                label, fontsize=8, ha="center", va="bottom",
                color="#2c2c2a",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#ffffff", edgecolor="#888", linewidth=0.5,
                    alpha=0.9,
                ),
                zorder=25,
            )
        else:
            # Virtual fixed point: small open marker in the chamber's color.
            col = chamber_color(fp["sigma"])
            ax.scatter(
                pt[0], pt[1], pt[2],
                marker="x", s=55,
                color=col, linewidth=1.3,
                depthshade=False, zorder=8, alpha=0.75,
            )

    # Axes labels / box
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")
    ax.set_xlim(0, box_max)
    ax.set_ylim(0, box_max)
    ax.set_zlim(0, box_max)
    ax.view_init(elev=22, azim=38)

    # Legend of chambers actually visited
    visited = sorted(set(bitmask_to_tuple(int(m), n) for m in sigma_trace))
    handles = [
        Line2D([0], [0], color=chamber_color(s), lw=3,
               label=f"σ = {sigma_label(_mask_from_tuple(s), n)}")
        for s in visited
    ]
    handles.append(
        Line2D([0], [0], marker="o", linestyle="",
               color="#d85a30", markeredgecolor="#4a1b0c",
               markersize=7, label="chamber crossing"),
    )
    handles.append(
        Line2D([0], [0], marker="X", linestyle="",
               color="#2c2c2a", markeredgecolor="#ffffff",
               markersize=11, label="true CTLN fixed point"),
    )
    handles.append(
        Line2D([0], [0], marker="x", linestyle="",
               color="#888", markersize=8, mew=1.3,
               label="virtual fixed point (outside its chamber)"),
    )
    handles.append(
        Line2D([0], [0], color="#888", linewidth=0.9, alpha=0.7,
               label="vector field on switching hyperplanes"),
    )
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=8,
        framealpha=0.9,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def _mask_from_tuple(sigma: tuple[int, ...]) -> int:
    m = 0
    for i in sigma:
        m |= (1 << i)
    return m


def _draw_timeseries_panel(ax, t, X, sigma_trace, crossings):
    n = X.shape[0]
    neuron_colors = ["#185fa5", "#3b6d11", "#993556"]

    # Shade by chamber in the background
    start = 0
    for c in crossings + [len(t) - 1]:
        s = bitmask_to_tuple(int(sigma_trace[start]), n)
        ax.axvspan(t[start], t[c], facecolor=chamber_color(s), alpha=0.22, linewidth=0)
        start = c

    # Traces on top
    for i in range(n):
        ax.plot(t, X[i], color=neuron_colors[i], linewidth=1.3, label=f"x_{i + 1}")

    # Dashed vertical lines at crossings
    for idx in crossings:
        ax.axvline(t[idx], color="#4a1b0c", linewidth=0.5, linestyle=":", alpha=0.6)

    ax.set_xlabel("time")
    ax.set_ylabel("firing rate")
    ax.set_title("Activity trace — background shading encodes active chamber σ(t)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_regime_panel(ax, W, theta, sigma_trace, crossings, n):
    """Summarize each distinct chamber visited: sigma, Jacobian eigenvalues."""
    # Get the sequence of chambers as they are visited (deduplicated consecutively)
    visited_order = []
    prev = None
    for m in sigma_trace:
        t = bitmask_to_tuple(int(m), n)
        if t != prev:
            visited_order.append(t)
            prev = t
    # Keep only the unique set for the table, but preserve first-visit order
    seen = []
    unique_visited = []
    for s in visited_order:
        if s not in seen:
            seen.append(s)
            unique_visited.append(s)

    ax.axis("off")
    ax.set_title(
        "Each chamber is a different linear system:\n"
        "same state space, different Jacobian, different dynamics",
        fontsize=10, loc="left",
    )

    # Build table rows
    headers = ["chamber σ", "color", "Jacobian eigenvalues", "behavior"]
    rows = []
    for s in unique_visited:
        reg = ChamberRegime(sigma=s, n=n, W=W, theta=theta)
        evs = reg.eigenvalues
        # Summarize eigenvalues as real/complex pairs
        ev_str = _format_eigs(evs)
        behavior = _classify_regime(evs)
        rows.append([
            f"σ = {sigma_label(_mask_from_tuple(s), n)}",
            chamber_color(s),
            ev_str,
            behavior,
        ])

    # Render as a manual "table" using ax.text so we can color the chamber swatches
    y = 0.88
    dy = 0.13
    col_x = [0.02, 0.22, 0.34, 0.78]

    # Header
    for x, h in zip(col_x, headers):
        ax.text(x, y, h, fontsize=9, fontweight="bold",
                transform=ax.transAxes, va="top")
    y -= 0.07
    ax.plot([0.02, 0.98], [y + 0.02, y + 0.02],
            transform=ax.transAxes, color="#888", linewidth=0.5)

    for row in rows:
        label, color, ev_str, behavior = row
        ax.text(col_x[0], y, label, fontsize=9, transform=ax.transAxes, va="top",
                family="monospace")
        # Color swatch
        sw = Rectangle(
            (col_x[1], y - 0.04), 0.08, 0.05,
            transform=ax.transAxes, facecolor=color, edgecolor="#444",
            linewidth=0.5,
        )
        ax.add_patch(sw)
        ax.text(col_x[2], y, ev_str, fontsize=8, transform=ax.transAxes, va="top",
                family="monospace")
        ax.text(col_x[3], y, behavior, fontsize=8, transform=ax.transAxes, va="top",
                style="italic")
        y -= dy


def _format_eigs(evs: np.ndarray, tol: float = 1e-8) -> str:
    parts = []
    seen = set()
    for ev in evs:
        # Deduplicate complex conjugate pairs visually
        key = (round(ev.real, 3), abs(round(ev.imag, 3)))
        if key in seen:
            continue
        seen.add(key)
        if abs(ev.imag) < tol:
            parts.append(f"{ev.real:+.2f}")
        else:
            parts.append(f"{ev.real:+.2f}±{abs(ev.imag):.2f}i")
    return ", ".join(parts)


def _classify_regime(evs: np.ndarray) -> str:
    reals = evs.real
    imags = np.abs(evs.imag)
    has_complex = np.any(imags > 1e-8)
    max_real = reals.max()
    if has_complex:
        if max_real > 1e-6:
            return "unstable spiral (expanding)"
        elif max_real < -1e-6:
            return "stable spiral (contracting)"
        else:
            return "neutral center"
    else:
        if max_real > 1e-6 and reals.min() < -1e-6:
            return "saddle"
        elif max_real > 1e-6:
            return "unstable node"
        else:
            return "stable node"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build_3cycle() -> np.ndarray:
    """
    Directed 3-cycle: 1 -> 2 -> 3 -> 1.
    adj[i, j] = 1 iff j -> i.
    """
    adj = np.array([
        [0, 0, 1],  # row 1: edge 3 -> 1
        [1, 0, 0],  # row 2: edge 1 -> 2
        [0, 1, 0],  # row 3: edge 2 -> 3
    ])
    return adj


def main():
    # ---- Parameters (legal CTLN range: delta > 0, theta > 0, 0 < eps < delta/(delta+1)) ----
    eps = 0.25
    delta = 0.5
    theta = 1.0

    adj = build_3cycle()
    W = build_ctln_W(adj, eps, delta)

    # Seed off the diagonal to land in a generic chamber
    x0 = np.array([0.1, 0.2, 0.05])
    t, X = simulate_ctln(W, theta, x0, t_span=(0.0, 80.0), t_eval_n=5000)

    # Trim off the transient so we're looking at the limit cycle itself
    start_idx = int(0.6 * len(t))
    t = t[start_idx:]
    X = X[:, start_idx:]
    t = t - t[0]

    sigma_trace = active_set_trace(X, W, theta)
    crossings = detect_crossings(sigma_trace)

    # Summary to stdout for sanity
    print(f"W =\n{W}")
    print(f"theta = {theta}")
    print(f"Trajectory length: {len(t)} samples over t ∈ [{t[0]:.2f}, {t[-1]:.2f}]")
    print(f"Detected {len(crossings)} chamber crossings")
    unique = sorted(set(sigma_trace.tolist()))
    n = W.shape[0]
    for m in unique:
        s = bitmask_to_tuple(int(m), n)
        print(f"  chamber visited: σ = {sigma_label(int(m), n)}")

    # Fixed point enumeration
    print("\nFixed point enumeration:")
    fps = all_fixed_points(W, theta)
    for fp in fps:
        marker = "★ TRUE" if fp["in_chamber"] else "  virtual"
        label = sigma_label(_mask_from_tuple(fp["sigma"]), n)
        pt_str = np.array2string(fp["point"], precision=3, suppress_small=True)
        print(f"  {marker}  σ={label:>10}  at x={pt_str}  ({fp['stability']})")

    fig = render_chamber_figure(
        W, theta, t, X, sigma_trace, crossings,
        title=f"3-cycle CTLN   (ε={eps}, δ={delta}, θ={theta}):   "
              f"limit cycle threading {len(set(sigma_trace.tolist()))} linear regimes",
        savepath="imgs/ctln_chamber_viz.png",
    )
    plt.show()
    return fig


if __name__ == "__main__":
    main()