#!/usr/bin/env bash
#SBATCH -A sds_baek_energetic
#SBATCH -J gnn_sine_baseline
#SBATCH -o gnn_sine_baseline.out
#SBATCH -e gnn_sine_baseline.err
#SBATCH -p standard
#SBATCH -t 01:00:00
#SBATCH -c 4
#SBATCH --mem=8G

echo "================================================================"
echo "STANDARD GNN BASELINE — 3-sine-wave toy problem (gCTLN contrast)"
echo "================================================================"
echo ""
echo "Architecture:"
echo "  (1) FeedforwardGNN — t as input, no temporal recurrence"
echo "  (2) RecurrentGNN   — learned neural ODE on 3-cycle graph"
echo "  Shared MPLayer: MLP(concat src,dst) → sum-aggregate → MLP update"
echo ""
echo "Task: fit three non-negative sines at 120° phase offsets"
echo "      x_i(t) = 0.4·sin(t + i·2π/3) + 0.5   on a 3-cycle graph"
echo ""
echo "Purpose: show vanilla GNNs struggle with sustained oscillations,"
echo "         exposing the inductive-bias advantage of gCTLNs."
echo "================================================================"

module purge
module load apptainer

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ============================================================
# PATHS
# ============================================================
SCRIPT_PATH="scripts/gnn_baseline.py"
OUTPUT_DIR="/scratch/xfd3tf/gnn_sine_baseline"

CONTAINER="/share/resources/containers/apptainer/pytorch-2.7.0.sif"

mkdir -p "$OUTPUT_DIR"

export GNN_SINE_OUTDIR="$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Script:   $SCRIPT_PATH"
echo "  Output:   $OUTPUT_DIR"
echo "  Device:   CPU (problem is tiny — 3 nodes, 1k timesteps)"
echo "  Threads:  $OMP_NUM_THREADS"
echo "================================================================"

apptainer run "$CONTAINER" "$SCRIPT_PATH"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training complete!"
    echo "   Plots saved to: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR"
else
    echo "❌ Training failed"
fi

exit $EXIT_CODE
