import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from functools import partial

# ─── 1. CORE GCTLN DYNAMICS (PURE JAX) ────────────────────────────────────────

@jax.jit
def build_W(edge_raw, noedge_raw, A):
    """Constructs the weight matrix enforcing gCTLN rules."""
    sp = jax.nn.softplus
    edge_weights = -1.0 + sp(edge_raw)
    noedge_weights = -1.0 - sp(noedge_raw)
    
    W = A * edge_weights + (1 - A) * noedge_weights
    I = jnp.eye(W.shape[0])
    return W * (1 - I) # Zero diagonal

@jax.jit
def gctln_dx_dt(x, edge_raw, noedge_raw, theta_raw, A):
    """The strict gCTLN derivative with absolute ReLU."""
    W = build_W(edge_raw, noedge_raw, A)
    theta = jnp.clip(jax.nn.softplus(theta_raw), 0.01, 1.0)
    
    drive = jnp.dot(W, x) + theta
    # Strict ReLU is required for the topological bounds of the attractors
    return -x + jnp.maximum(0.0, drive)

@partial(jax.jit, static_argnames=("T", "dt"))
def simulate_euler(x0, edge_raw, noedge_raw, theta_raw, A, T=10.0, dt=0.05):
    steps = int(T / dt)
    
    def step_fn(x, _):
        dx = gctln_dx_dt(x, edge_raw, noedge_raw, theta_raw, A)
        x_next = x + dx * dt
        return x_next, x_next

    _, trajectory = jax.lax.scan(step_fn, x0, jnp.arange(steps))
    return trajectory

# ─── 2. THE FITNESS FUNCTION (REPLACING SINE WAVES) ───────────────────────────

@jax.jit
def target_vector_field(x):
    """
    Defines a stable limit cycle centered in the positive orthant.
    """
    # 1. Shift the center of field to positive space
    center = jnp.array([0.5, 0.5, 0.5])
    x_c = x - center
    
    # 2. Pure Rotation (Cyclic Flow 0 -> 1 -> 2 -> 0)
    rotation = jnp.array([
        [ 0.0, -1.0,  1.0],
        [ 1.0,  0.0, -1.0],
        [-1.0,  1.0,  0.0]
    ])
    v_rot = jnp.dot(rotation, x_c)
    
    # 3. Radial Rubber Band (Attract to a specific radius R)
    R = 0.3  # We want oscillations that swing between 0.2 and 0.8
    r_sq = jnp.sum(x_c**2)
    
    # If r_sq < R^2 (too close to center), this is positive (pushes out).
    # If r_sq > R^2 (too far away), this is negative (pulls in).
    v_rad = (R**2 - r_sq) * x_c  
    
    # Combine rotation and radial correction
    return v_rot + 1.0 * v_rad

@jax.jit
def evaluate_fitness(flat_params, x0, A, param_shapes):
    """
    Unflattens parameters, runs the simulation, and calculates fitness.
    Higher fitness is better.
    """
    n = A.shape[0]
    
    # Unpack flat vector into original parameter matrices
    edge_raw = flat_params[0:n*n].reshape(n, n)
    noedge_raw = flat_params[n*n : 2*n*n].reshape(n, n)
    theta_raw = flat_params[2*n*n : 2*n*n + n]
    
    traj = simulate_euler(x0, edge_raw, noedge_raw, theta_raw, A, T=50.0, dt=0.05)
    
    # 1. Trajectory Target: Sustained Variance
    # We want the second half of the simulation to still be oscillating
    half = traj.shape[0] // 2
    late_traj = traj[half:]
    
    variance_score = jnp.mean(jnp.var(late_traj, axis=0))
    decay_penalty = jnp.maximum(0.0, 0.05 - jnp.mean(late_traj))**2
    
    # 2. Vector Field Target: Cosine similarity to a target cycle
    # Sample points from the trajectory to see if the "wind" aligns
    sample_points = late_traj[::10] 
    
    def align_fn(x):
        v_model = gctln_dx_dt(x, edge_raw, noedge_raw, theta_raw, A)
        v_tgt = target_vector_field(x)
        
        # Cosine similarity
        norm_m = jnp.linalg.norm(v_model) + 1e-6
        norm_t = jnp.linalg.norm(v_tgt) + 1e-6
        return jnp.dot(v_model, v_tgt) / (norm_m * norm_t)

    alignment_score = jnp.mean(jax.vmap(align_fn)(sample_points))
    
    # Combine: Reward variance and alignment, heavily penalize decaying to 0
    fitness = (10.0 * variance_score) + (5.0 * alignment_score) - (100.0 * decay_penalty)
    return fitness

# ─── 3. OPENAI-ES OPTIMIZATION LOOP ───────────────────────────────────────────

def train_es(A, n_epochs=500, pop_size=256, sigma=0.1, alpha=0.05):
    """
    OpenAI Evolution Strategies. Scales at O(N) instead of O(N^2) like CMA-ES.
    """
    n = A.shape[0]
    key = jax.random.PRNGKey(42)
    
    # Initialize flat parameters (edge, noedge, theta)
    n_params = (n * n) + (n * n) + n
    params = jnp.full(n_params, -2.0) 
    params = params.at[-n:].set(-1.84) # Initial theta raw
    
    x0 = jnp.array([0.3, 0.2, 0.1])
    
    # Vectorize the fitness evaluation to run the whole population at once
    vmap_fitness = jax.jit(jax.vmap(evaluate_fitness, in_axes=(0, None, None, None)))
    
    history = []
    
    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        
        # 1. Generate Population Noise
        noise = jax.random.normal(subkey, (pop_size, n_params))
        
        # Antithetic Sampling: mirror the perturbations to reduce variance
        noise = jnp.concatenate([noise, -noise], axis=0) 
        population = params + sigma * noise
        
        # 2. Evaluate entire population in parallel (GPU/CPU)
        fitnesses = vmap_fitness(population, x0, A, None)
        
        # 3. Rank or Standardize Fitness (crucial for stability)
        fit_mean = jnp.mean(fitnesses)
        fit_std = jnp.std(fitnesses) + 1e-8
        normalized_fitness = (fitnesses - fit_mean) / fit_std
        
        # 4. Gradient Estimation & Parameter Update
        # grad = 1/(N*sigma) * sum(fitness * noise)
        grad_est = jnp.mean(normalized_fitness[:, None] * noise, axis=0) / sigma
        
        # Update using basic SGD (can be wrapped in Adam via Optax if desired)
        params = params + alpha * grad_est
        
        history.append(fit_mean.item())
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Mean Fitness: {fit_mean:.4f} | Max Fit: {jnp.max(fitnesses):.4f}")
            
    return params, history

# ─── 4. EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Define adjacency for a 3-node cyclic motif
    # (1 -> 2 -> 3 -> 1)
    A = jnp.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    
    print("Starting Highly Parallel ES Training via JAX...")
    learned_params, hist = train_es(A, n_epochs=300, pop_size=512)
    
    # Simulate final result
    n = A.shape[0]
    e_raw = learned_params[0:n*n].reshape(n, n)
    ne_raw = learned_params[n*n : 2*n*n].reshape(n, n)
    t_raw = learned_params[-n:]
    
    final_traj = simulate_euler(jnp.array([0.3, 0.2, 0.1]), e_raw, ne_raw, t_raw, A, T=100.0)
    
    plt.figure(figsize=(10, 4))
    plt.plot(final_traj)
    plt.title("Learned gCTLN Dynamics (Strict ReLU)")
    plt.xlabel("Time steps")
    plt.ylabel("Activity")
    plt.savefig("imgs/jaxgctlnout.png")
