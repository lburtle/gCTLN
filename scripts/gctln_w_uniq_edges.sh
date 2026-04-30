#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gctln_smooth
#SBATCH -o gctln_smooth.out
#SBATCH -e gctln_smooth.err
#SBATCH -p standard
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=8G

echo "================================================================"
echo "gCTLN w/ SOFTPLUS SMOOTHING — 3-sine-wave toy problem"
echo "================================================================"
echo ""
echo "Architecture:"
echo "  GCTLNPerEdge with per-edge weight learning"
echo "  ODE: dx/dt = -x + softplus(W·x + θ, β)"
echo "       (softplus replaces torch.clamp to smooth gradients"
echo "        through the adjoint method)"
echo ""
echo "Task: fit three non-negative sines at 120° phase offsets"
echo "      x_i(t) = 0.4·sin(t + i·2π/3) + 0.5   on a 3-cycle motif"
echo ""
echo "Purpose: test whether smoothed CTLN dynamics enable"
echo "         gradient-based discovery of oscillatory regimes"
echo "         (previously failed under piecewise-linear ReLU)."
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ============================================================
# PATHS
# ============================================================
SCRIPT_PATH="scripts/gctln_w_uniq_edges.py"
OUTPUT_DIR="/scratch/xfd3tf/gctln_smooth"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

mkdir -p "$OUTPUT_DIR"
mkdir -p "imgs"   # script writes to imgs/ by default

echo ""
echo "Configuration:"
echo "  Script:   $SCRIPT_PATH"
echo "  Output:   $OUTPUT_DIR"
echo "  Device:   CPU (3-neuron CTLN, 1k timesteps)"
echo "  Threads:  $OMP_NUM_THREADS"
echo "================================================================"

apptainer run "$CONTAINER" "$SCRIPT_PATH"

apptainer exec "$CONTAINER" bash -c "
    source ~/venvs/gctln-container/bin/activate && \
    python $SCRIPT_PATH
"

EXIT_CODE=$?

# Copy generated plots to scratch so they persist
if [ -d "imgs" ]; then
    cp -v imgs/gctln_*.png "$OUTPUT_DIR/" 2>/dev/null || true
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
    echo "   Plots saved to: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
else
    echo "❌ Training failed"
fi

exit $EXIT_CODE
