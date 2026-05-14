#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gctln_jax_es
#SBATCH -o gctln_jax_es.out
#SBATCH -e gctln_jax_es.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=32G

echo "================================================================"
echo "gCTLN w/ STRICT RELU & JAX OPENAI-ES — 3-node Vector Field Target"
echo "================================================================"
echo ""
echo "Architecture:"
echo "  Pure JAX implementation of gCTLN with per-edge weights"
echo "  ODE: dx/dt = -x + ReLU(W·x + θ)"
echo "       (Restored strict ReLU bounds, optimized via Evolution Strategies)"
echo ""
echo "Task: Find topological limit cycles using a phase-space vector"
echo "      field target, rather than time-series sine matching."
echo ""
echo "Purpose: Use jax.vmap to evaluate thousands of candidate networks"
echo "         in parallel on the GPU to bypass shattered gradients."
echo "================================================================"

module purge
module load apptainer

# JAX optimization flags
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

# ============================================================
# PATHS
# ============================================================
SCRIPT_PATH="scripts/jaxgctln.py"
OUTPUT_DIR="/scratch/xfd3tf/gctln_jax_es"

# You can stick with the PyTorch container if your venv has JAX installed inside it, 
# or switch to a native JAX container if Rivanna provides one.
CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

mkdir -p "$OUTPUT_DIR"
mkdir -p "imgs"

echo ""
echo "Configuration:"
echo "  Script:   $SCRIPT_PATH"
echo "  Output:   $OUTPUT_DIR"
echo "  Device:   GPU 1x (vmapped ES Population)"
echo "================================================================"

# Execute purely through the virtual environment mapping
apptainer exec --nv "$CONTAINER" bash -c "
    source ~/venvs/gctln-container/bin/activate && \
    python $SCRIPT_PATH
"

EXIT_CODE=$?

if [ -d "imgs" ]; then
    cp -v imgs/gctln_*.png "$OUTPUT_DIR/" 2>/dev/null || true
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ES Training complete!"
    echo "   Plots saved to: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
else
    echo "❌ Training failed"
fi

exit $EXIT_CODE
